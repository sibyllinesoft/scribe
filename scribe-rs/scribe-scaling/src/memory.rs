//! Memory management and pooling for efficient resource utilization.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub pool_size: usize,
    pub max_allocation: usize,
    pub enable_monitoring: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size: 1024,
            max_allocation: 100 * 1024 * 1024, // 100MB
            enable_monitoring: true,
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub allocations: u64,
    pub deallocations: u64,
    pub current_usage: usize,
    pub peak_usage: usize,
}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    config: MemoryConfig,
    available: Arc<Mutex<VecDeque<Vec<u8>>>>,
    stats: Arc<Mutex<MemoryStats>>,
}

impl MemoryPool {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            available: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(MemoryStats::default())),
        }
    }

    pub fn allocate(&self, size: usize) -> Vec<u8> {
        let mut buffer = self
            .available
            .lock()
            .pop_front()
            .unwrap_or_else(|| Vec::with_capacity(size));

        buffer.resize(size, 0);

        if self.config.enable_monitoring {
            let mut stats = self.stats.lock();
            stats.allocations += 1;
            stats.current_usage += size;
            stats.peak_usage = stats.peak_usage.max(stats.current_usage);
        }

        buffer
    }

    pub fn deallocate(&self, mut buffer: Vec<u8>) {
        let buffer_len = buffer.len();

        if buffer.capacity() <= self.config.max_allocation {
            buffer.clear();
            self.available.lock().push_back(buffer);
        }

        if self.config.enable_monitoring {
            let mut stats = self.stats.lock();
            stats.deallocations += 1;
            stats.current_usage = stats.current_usage.saturating_sub(buffer_len);
        }
    }

    pub fn get_stats(&self) -> MemoryStats {
        self.stats.lock().clone()
    }
}

/// Memory manager for the scaling system
pub struct MemoryManager {
    pool: MemoryPool,
}

impl MemoryManager {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            pool: MemoryPool::new(config),
        }
    }

    pub fn allocate(&self, size: usize) -> Vec<u8> {
        self.pool.allocate(size)
    }

    pub fn deallocate(&self, buffer: Vec<u8>) {
        self.pool.deallocate(buffer);
    }

    pub fn get_stats(&self) -> MemoryStats {
        self.pool.get_stats()
    }
}
