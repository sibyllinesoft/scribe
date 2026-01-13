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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.pool_size, 1024);
        assert_eq!(config.max_allocation, 100 * 1024 * 1024);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_memory_config_custom() {
        let config = MemoryConfig {
            pool_size: 512,
            max_allocation: 50 * 1024 * 1024,
            enable_monitoring: false,
        };
        assert_eq!(config.pool_size, 512);
        assert_eq!(config.max_allocation, 50 * 1024 * 1024);
        assert!(!config.enable_monitoring);
    }

    #[test]
    fn test_memory_config_serialize() {
        let config = MemoryConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: MemoryConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.pool_size, deserialized.pool_size);
        assert_eq!(config.enable_monitoring, deserialized.enable_monitoring);
    }

    #[test]
    fn test_memory_stats_default() {
        let stats = MemoryStats::default();
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.deallocations, 0);
        assert_eq!(stats.current_usage, 0);
        assert_eq!(stats.peak_usage, 0);
    }

    #[test]
    fn test_memory_stats_serialize() {
        let stats = MemoryStats {
            allocations: 10,
            deallocations: 5,
            current_usage: 1024,
            peak_usage: 2048,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: MemoryStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats.allocations, deserialized.allocations);
        assert_eq!(stats.peak_usage, deserialized.peak_usage);
    }

    #[test]
    fn test_memory_pool_new() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);
        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 0);
    }

    #[test]
    fn test_memory_pool_allocate() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let buffer = pool.allocate(1024);
        assert_eq!(buffer.len(), 1024);

        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.current_usage, 1024);
    }

    #[test]
    fn test_memory_pool_deallocate() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let buffer = pool.allocate(1024);
        pool.deallocate(buffer);

        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.deallocations, 1);
        assert_eq!(stats.current_usage, 0);
    }

    #[test]
    fn test_memory_pool_reuse() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        // Allocate and deallocate
        let buffer1 = pool.allocate(1024);
        pool.deallocate(buffer1);

        // Second allocation should reuse from pool
        let buffer2 = pool.allocate(512);
        assert_eq!(buffer2.len(), 512);

        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 2);
    }

    #[test]
    fn test_memory_pool_peak_usage() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        // Allocate several buffers
        let buf1 = pool.allocate(1000);
        let buf2 = pool.allocate(2000);

        let stats = pool.get_stats();
        assert_eq!(stats.peak_usage, 3000);

        // Deallocate one
        pool.deallocate(buf1);

        let stats = pool.get_stats();
        // Peak should still be 3000
        assert_eq!(stats.peak_usage, 3000);
        assert_eq!(stats.current_usage, 2000);

        pool.deallocate(buf2);
    }

    #[test]
    fn test_memory_pool_monitoring_disabled() {
        let config = MemoryConfig {
            pool_size: 1024,
            max_allocation: 100 * 1024 * 1024,
            enable_monitoring: false,
        };
        let pool = MemoryPool::new(config);

        let buffer = pool.allocate(1024);
        pool.deallocate(buffer);

        let stats = pool.get_stats();
        // With monitoring disabled, stats should remain at 0
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.deallocations, 0);
    }

    #[test]
    fn test_memory_manager_new() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config);
        let stats = manager.get_stats();
        assert_eq!(stats.allocations, 0);
    }

    #[test]
    fn test_memory_manager_allocate_deallocate() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config);

        let buffer = manager.allocate(2048);
        assert_eq!(buffer.len(), 2048);

        manager.deallocate(buffer);

        let stats = manager.get_stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.deallocations, 1);
    }

    #[test]
    fn test_memory_pool_large_allocation() {
        let config = MemoryConfig {
            pool_size: 1024,
            max_allocation: 100, // Very small max
            enable_monitoring: true,
        };
        let pool = MemoryPool::new(config);

        // Allocate buffer larger than max_allocation
        let buffer = pool.allocate(200);
        assert_eq!(buffer.len(), 200);

        // Deallocate - should not be pooled due to size
        pool.deallocate(buffer);

        // Stats should still update
        let stats = pool.get_stats();
        assert_eq!(stats.deallocations, 1);
    }
}
