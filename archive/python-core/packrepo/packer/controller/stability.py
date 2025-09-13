"""
V3 Stability Tracking System

Implements oscillation detection, epoch ban list management, and stability
metrics tracking for the V3 demotion controller as specified in TODO.md.

Key Features:
- Oscillation detection across selection runs
- Epoch-based ban list to prevent re-promotion
- Stability metrics computation and tracking
- Event logging for detailed analysis
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Optional, Any, Tuple
from collections import defaultdict, deque
import logging

from ..selector.base import SelectionResult

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of stability tracking events."""
    
    SELECTION = "selection"        # Normal selection
    DEMOTION = "demotion"         # Chunk demoted
    PROMOTION = "promotion"       # Chunk promoted
    OSCILLATION = "oscillation"   # Oscillation detected
    BAN = "ban"                   # Chunk banned
    UNBAN = "unban"              # Chunk unbanned


@dataclass
class OscillationEvent:
    """
    Event representing a detected oscillation or stability-related action.
    
    Tracks changes in chunk selection modes and promotion/demotion patterns
    to detect and prevent oscillation behaviors.
    """
    
    chunk_id: str                    # Chunk involved in event
    epoch: int                       # Epoch when event occurred
    event_type: EventType           # Type of stability event
    
    # Mode information
    old_mode: Optional[str] = None   # Previous chunk mode
    new_mode: Optional[str] = None   # New chunk mode after event
    
    # Context information
    strategy: Optional[str] = None   # Strategy that caused event
    risk_score: float = 0.0         # Oscillation risk score (0-1)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate event consistency."""
        # Get event type value (handle both enum and string)
        event_type_str = self.event_type.value if hasattr(self.event_type, 'value') else self.event_type
        
        if event_type_str in ['demotion', 'promotion']:
            if self.old_mode == self.new_mode:
                logger.warning(
                    f"Mode change event {event_type_str} for {self.chunk_id} "
                    f"has same old and new modes: {self.old_mode}"
                )
    
    def is_mode_change(self) -> bool:
        """Check if this event represents a mode change."""
        event_type_str = self.event_type.value if hasattr(self.event_type, 'value') else self.event_type
        return (
            event_type_str in ['demotion', 'promotion'] and
            self.old_mode != self.new_mode
        )
    
    def get_mode_change_direction(self) -> Optional[str]:
        """Get direction of mode change if applicable."""
        if not self.is_mode_change():
            return None
            
        mode_order = {'signature': 0, 'summary': 1, 'full': 2}
        old_level = mode_order.get(self.old_mode, -1)
        new_level = mode_order.get(self.new_mode, -1)
        
        if old_level < new_level:
            return 'promotion'
        elif old_level > new_level:
            return 'demotion'
        else:
            return 'lateral'


@dataclass
class EpochBanList:
    """
    Epoch-based ban list to prevent oscillating chunk re-promotion.
    
    Maintains banned chunks with expiration epochs to prevent
    oscillations within and across epochs.
    """
    
    current_epoch: int = 0
    banned_chunks: Dict[str, int] = field(default_factory=dict)  # chunk_id -> ban_until_epoch
    ban_history: List[Tuple[str, int, int]] = field(default_factory=list)  # (chunk_id, banned_at, expires_at)
    
    def add_chunk(self, chunk_id: str, ban_until_epoch: int):
        """Add chunk to ban list until specified epoch."""
        if ban_until_epoch <= self.current_epoch:
            logger.warning(
                f"Cannot ban {chunk_id} until epoch {ban_until_epoch} "
                f"(current epoch: {self.current_epoch})"
            )
            return
            
        old_ban = self.banned_chunks.get(chunk_id)
        self.banned_chunks[chunk_id] = ban_until_epoch
        
        # Record in history
        self.ban_history.append((chunk_id, self.current_epoch, ban_until_epoch))
        
        if old_ban:
            logger.info(
                f"Extended ban for {chunk_id} from epoch {old_ban} to {ban_until_epoch}"
            )
        else:
            logger.info(f"Banned {chunk_id} until epoch {ban_until_epoch}")
    
    def is_banned(self, chunk_id: str) -> bool:
        """Check if chunk is currently banned."""
        ban_until = self.banned_chunks.get(chunk_id)
        return ban_until is not None and ban_until > self.current_epoch
    
    def get_banned_chunks(self) -> Dict[str, int]:
        """Get all currently banned chunks with their expiration epochs."""
        return {
            chunk_id: ban_until
            for chunk_id, ban_until in self.banned_chunks.items()
            if ban_until > self.current_epoch
        }
    
    def advance_epoch(self, new_epoch: int) -> List[str]:
        """
        Advance to new epoch and return list of chunks that became unbanned.
        
        Args:
            new_epoch: New epoch number
            
        Returns:
            List of chunk IDs that became unbanned
        """
        if new_epoch <= self.current_epoch:
            logger.warning(f"Cannot advance epoch backwards: {new_epoch} <= {self.current_epoch}")
            return []
        
        old_epoch = self.current_epoch
        self.current_epoch = new_epoch
        
        # Find chunks that became unbanned
        unbanned = []
        for chunk_id, ban_until in list(self.banned_chunks.items()):
            if ban_until <= new_epoch:
                unbanned.append(chunk_id)
                del self.banned_chunks[chunk_id]
        
        if unbanned:
            logger.info(
                f"Epoch {old_epoch} -> {new_epoch}: Unbanned {len(unbanned)} chunks: {unbanned}"
            )
        
        return unbanned
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ban list statistics."""
        currently_banned = self.get_banned_chunks()
        
        return {
            'current_epoch': self.current_epoch,
            'currently_banned_count': len(currently_banned),
            'total_bans_issued': len(self.ban_history),
            'currently_banned_chunks': list(currently_banned.keys()),
            'average_ban_duration': self._calculate_average_ban_duration(),
            'ban_history_count': len(self.ban_history),
        }
    
    def _calculate_average_ban_duration(self) -> float:
        """Calculate average ban duration from history."""
        if not self.ban_history:
            return 0.0
        
        total_duration = sum(
            expires_at - banned_at
            for chunk_id, banned_at, expires_at in self.ban_history
        )
        
        return total_duration / len(self.ban_history)


class StabilityTracker:
    """
    Comprehensive stability tracking system for V3 demotion controller.
    
    Tracks oscillations, manages epoch ban lists, and provides stability
    metrics to prevent selection instability and performance degradation.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize stability tracker.
        
        Args:
            max_history_size: Maximum number of events to keep in history
        """
        self.max_history_size = max_history_size
        
        # Core tracking state
        self.ban_list = EpochBanList()
        self.event_history: deque[OscillationEvent] = deque(maxlen=max_history_size)
        
        # Oscillation detection
        self.chunk_mode_history: Dict[str, List[Tuple[int, str]]] = defaultdict(list)  # chunk_id -> [(epoch, mode)]
        self.oscillation_counts: Dict[str, int] = defaultdict(int)
        
        # Performance tracking
        self.total_oscillations = 0
        self.prevented_oscillations = 0
        self.stability_score = 1.0
        
        # Analysis cache
        self._risk_cache: Dict[str, float] = {}
        self._cache_epoch = -1
    
    def record_event(self, event: OscillationEvent):
        """
        Record a stability-related event.
        
        Args:
            event: Event to record
        """
        self.event_history.append(event)
        
        # Update mode history for oscillation detection
        if event.is_mode_change():
            self.chunk_mode_history[event.chunk_id].append((event.epoch, event.new_mode))
            
            # Keep only recent history to prevent memory growth
            if len(self.chunk_mode_history[event.chunk_id]) > 10:
                self.chunk_mode_history[event.chunk_id] = \
                    self.chunk_mode_history[event.chunk_id][-10:]
        
        # Update oscillation counts
        if event.event_type == EventType.OSCILLATION:
            self.oscillation_counts[event.chunk_id] += 1
            self.total_oscillations += 1
        
        # Clear cache when new events are recorded
        if event.epoch != self._cache_epoch:
            self._risk_cache.clear()
            self._cache_epoch = event.epoch
        
        event_type_str = event.event_type.value if hasattr(event.event_type, 'value') else event.event_type
        logger.debug(f"Recorded stability event: {event_type_str} for {event.chunk_id}")
    
    def detect_oscillations(
        self,
        current_selection: SelectionResult,
        previous_selections: List[SelectionResult],
        look_back_epochs: int = 5
    ) -> List[OscillationEvent]:
        """
        Detect oscillations by comparing current selection with previous selections.
        
        Args:
            current_selection: Current selection result
            previous_selections: Previous selection results for comparison
            look_back_epochs: Number of previous epochs to analyze
            
        Returns:
            List of detected oscillation events
        """
        if not previous_selections:
            return []
        
        oscillations = []
        current_epoch = self.ban_list.current_epoch
        
        # Analyze recent selections for oscillation patterns
        recent_selections = previous_selections[-look_back_epochs:]
        
        # Track mode changes for each chunk
        chunk_changes = defaultdict(list)
        
        # Build change history from previous selections
        for i, selection in enumerate(recent_selections):
            epoch = current_epoch - len(recent_selections) + i
            for chunk in selection.selected_chunks:
                mode = selection.chunk_modes.get(chunk.id, 'full')
                chunk_changes[chunk.id].append((epoch, mode))
        
        # Add current selection
        for chunk in current_selection.selected_chunks:
            mode = current_selection.chunk_modes.get(chunk.id, 'full')
            chunk_changes[chunk.id].append((current_epoch, mode))
        
        # Detect oscillations
        for chunk_id, changes in chunk_changes.items():
            if len(changes) < 3:  # Need at least 3 points to detect oscillation
                continue
            
            # Look for A->B->A patterns (oscillation)
            for i in range(len(changes) - 2):
                epoch1, mode1 = changes[i]
                epoch2, mode2 = changes[i + 1]
                epoch3, mode3 = changes[i + 2]
                
                # Check for oscillation pattern
                if mode1 == mode3 and mode1 != mode2:
                    # Found A->B->A oscillation
                    risk_score = self.calculate_oscillation_risk(chunk_id, current_epoch)
                    
                    oscillation = OscillationEvent(
                        chunk_id=chunk_id,
                        epoch=current_epoch,
                        event_type=EventType.OSCILLATION,
                        old_mode=mode2,
                        new_mode=mode3,
                        risk_score=risk_score,
                        metadata={
                            'pattern': f"{mode1}->{mode2}->{mode3}",
                            'epochs': [epoch1, epoch2, epoch3],
                            'oscillation_count': self.oscillation_counts[chunk_id] + 1,
                        }
                    )
                    
                    oscillations.append(oscillation)
                    logger.warning(
                        f"Detected oscillation for {chunk_id}: {mode1}->{mode2}->{mode3} "
                        f"across epochs {epoch1}, {epoch2}, {epoch3}"
                    )
        
        # Record detected oscillations
        for osc in oscillations:
            self.record_event(osc)
        
        if oscillations:
            # Update stability score
            self._update_stability_score(len(oscillations))
        
        return oscillations
    
    def calculate_oscillation_risk(self, chunk_id: str, current_epoch: int) -> float:
        """
        Calculate oscillation risk for a chunk based on its history.
        
        Args:
            chunk_id: Chunk to analyze
            current_epoch: Current epoch
            
        Returns:
            Risk score between 0.0 (low risk) and 1.0 (high risk)
        """
        # Check cache first
        cache_key = f"{chunk_id}_{current_epoch}"
        if cache_key in self._risk_cache:
            return self._risk_cache[cache_key]
        
        # Calculate risk based on multiple factors
        base_risk = 0.0
        
        # Factor 1: Historical oscillation count
        oscillation_count = self.oscillation_counts.get(chunk_id, 0)
        if oscillation_count > 0:
            base_risk += min(0.4, oscillation_count * 0.1)
        
        # Factor 2: Recent mode change frequency
        mode_history = self.chunk_mode_history.get(chunk_id, [])
        recent_history = [
            (epoch, mode) for epoch, mode in mode_history
            if epoch >= current_epoch - 5  # Last 5 epochs
        ]
        
        if len(recent_history) >= 3:
            # Count mode changes in recent history
            mode_changes = 0
            for i in range(1, len(recent_history)):
                if recent_history[i][1] != recent_history[i-1][1]:
                    mode_changes += 1
            
            change_rate = mode_changes / (len(recent_history) - 1)
            base_risk += min(0.3, change_rate * 0.5)
        
        # Factor 3: Current ban status
        if self.ban_list.is_banned(chunk_id):
            base_risk += 0.3
        
        # Factor 4: Recent events for this chunk
        recent_events = [
            event for event in self.event_history
            if event.chunk_id == chunk_id and event.epoch >= current_epoch - 3
        ]
        
        if len(recent_events) >= 2:
            base_risk += min(0.2, len(recent_events) * 0.05)
        
        # Clamp to [0, 1] range
        risk_score = max(0.0, min(1.0, base_risk))
        
        # Cache result
        self._risk_cache[cache_key] = risk_score
        
        return risk_score
    
    def add_to_ban_list(self, chunk_id: str, ban_until_epoch: int):
        """Add chunk to ban list until specified epoch."""
        self.ban_list.add_chunk(chunk_id, ban_until_epoch)
        
        # Record ban event
        ban_event = OscillationEvent(
            chunk_id=chunk_id,
            epoch=self.ban_list.current_epoch,
            event_type=EventType.BAN,
            metadata={'ban_until': ban_until_epoch}
        )
        self.record_event(ban_event)
    
    def get_banned_chunks(self, epoch: Optional[int] = None) -> Dict[str, int]:
        """
        Get currently banned chunks.
        
        Args:
            epoch: Epoch to check (uses current if None)
            
        Returns:
            Dictionary mapping chunk IDs to their ban expiration epochs
        """
        if epoch is not None:
            # Temporarily set epoch for check
            old_epoch = self.ban_list.current_epoch
            self.ban_list.current_epoch = epoch
            result = self.ban_list.get_banned_chunks()
            self.ban_list.current_epoch = old_epoch
            return result
        else:
            return self.ban_list.get_banned_chunks()
    
    def advance_epoch(self, new_epoch: int):
        """
        Advance to new epoch and process ban list updates.
        
        Args:
            new_epoch: New epoch number
        """
        unbanned_chunks = self.ban_list.advance_epoch(new_epoch)
        
        # Record unban events
        for chunk_id in unbanned_chunks:
            unban_event = OscillationEvent(
                chunk_id=chunk_id,
                epoch=new_epoch,
                event_type=EventType.UNBAN,
                metadata={'unbanned_at_epoch': new_epoch}
            )
            self.record_event(unban_event)
        
        # Clear risk cache for new epoch
        self._risk_cache.clear()
        self._cache_epoch = new_epoch
        
        logger.info(f"Advanced to epoch {new_epoch}, unbanned {len(unbanned_chunks)} chunks")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive stability metrics."""
        current_epoch = self.ban_list.current_epoch
        
        # Calculate recent oscillation rate
        recent_events = [
            event for event in self.event_history
            if event.epoch >= current_epoch - 5  # Last 5 epochs
        ]
        
        recent_oscillations = [
            event for event in recent_events
            if event.event_type == EventType.OSCILLATION
        ]
        
        oscillation_rate = len(recent_oscillations) / max(1, len(recent_events))
        
        # Get ban list statistics
        ban_stats = self.ban_list.get_statistics()
        
        # Calculate chunk stability distribution
        chunk_risk_distribution = self._calculate_risk_distribution()
        
        return {
            # Core metrics
            'stability_score': self.stability_score,
            'total_oscillations': self.total_oscillations,
            'prevented_oscillations': self.prevented_oscillations,
            'current_epoch': current_epoch,
            
            # Rate metrics
            'oscillation_rate': oscillation_rate,
            'recent_oscillations': len(recent_oscillations),
            'recent_events': len(recent_events),
            
            # Ban list metrics
            'ban_list_stats': ban_stats,
            
            # Risk distribution
            'risk_distribution': chunk_risk_distribution,
            
            # History metrics
            'event_history_size': len(self.event_history),
            'tracked_chunks': len(self.chunk_mode_history),
            'chunks_with_oscillations': len([
                chunk_id for chunk_id, count in self.oscillation_counts.items()
                if count > 0
            ]),
        }
    
    def export_stability_report(self) -> Dict[str, Any]:
        """Export comprehensive stability analysis report."""
        metrics = self.get_metrics()
        current_epoch = self.ban_list.current_epoch
        
        # Detailed event analysis
        event_breakdown = defaultdict(int)
        events_by_epoch = defaultdict(list)
        
        for event in self.event_history:
            event_type_str = event.event_type.value if hasattr(event.event_type, 'value') else event.event_type
            event_breakdown[event_type_str] += 1
            events_by_epoch[event.epoch].append(event)
        
        # Top oscillating chunks
        top_oscillating = sorted(
            self.oscillation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Chunk risk analysis
        high_risk_chunks = []
        for chunk_id in self.chunk_mode_history.keys():
            risk = self.calculate_oscillation_risk(chunk_id, current_epoch)
            if risk > 0.5:
                high_risk_chunks.append((chunk_id, risk))
        
        high_risk_chunks.sort(key=lambda x: x[1], reverse=True)
        
        report = {
            'summary': metrics,
            'event_breakdown': dict(event_breakdown),
            'events_by_epoch': {
                epoch: len(events) for epoch, events in events_by_epoch.items()
            },
            'top_oscillating_chunks': top_oscillating,
            'high_risk_chunks': high_risk_chunks[:10],
            'recommendations': self._generate_recommendations(),
            'generated_at': time.time(),
        }
        
        return report
    
    def _update_stability_score(self, new_oscillations: int):
        """Update overall stability score based on new oscillations."""
        if new_oscillations > 0:
            # Decay stability score based on oscillation severity
            decay_factor = min(0.1, new_oscillations * 0.02)
            self.stability_score = max(0.0, self.stability_score - decay_factor)
        else:
            # Slowly recover stability score
            recovery_rate = 0.01
            self.stability_score = min(1.0, self.stability_score + recovery_rate)
    
    def _calculate_risk_distribution(self) -> Dict[str, int]:
        """Calculate distribution of chunks by risk level."""
        current_epoch = self.ban_list.current_epoch
        distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for chunk_id in self.chunk_mode_history.keys():
            risk = self.calculate_oscillation_risk(chunk_id, current_epoch)
            
            if risk <= 0.25:
                distribution['low'] += 1
            elif risk <= 0.5:
                distribution['medium'] += 1
            elif risk <= 0.75:
                distribution['high'] += 1
            else:
                distribution['critical'] += 1
        
        return distribution
    
    def _generate_recommendations(self) -> List[str]:
        """Generate stability improvement recommendations."""
        recommendations = []
        metrics = self.get_metrics()
        
        if metrics['oscillation_rate'] > 0.1:
            recommendations.append(
                "High oscillation rate detected. Consider increasing ban duration "
                "or tightening demotion thresholds."
            )
        
        if metrics['ban_list_stats']['currently_banned_count'] > 10:
            recommendations.append(
                "Many chunks currently banned. Review demotion strategies "
                "to prevent excessive banning."
            )
        
        if metrics['stability_score'] < 0.8:
            recommendations.append(
                "Low stability score. Increase corrective step budget "
                "or reduce demotion aggressiveness."
            )
        
        risk_dist = metrics['risk_distribution']
        high_risk_ratio = (risk_dist.get('high', 0) + risk_dist.get('critical', 0)) / max(1, sum(risk_dist.values()))
        
        if high_risk_ratio > 0.2:
            recommendations.append(
                "High proportion of high-risk chunks. Consider preemptive "
                "banning or more conservative selection strategies."
            )
        
        if not recommendations:
            recommendations.append("Stability metrics look good. Continue current approach.")
        
        return recommendations