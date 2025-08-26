"""
Golden smoke flow tests for PackRepo E2E validation.

Implements the golden smoke flows that validate core PackRepo functionality
with real repositories and realistic usage patterns.
"""

from __future__ import annotations

import pytest
import tempfile
import shutil
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from packrepo.packer.oracles.registry import OracleRegistry
from packrepo.packer.packfmt.base import PackFormat


@dataclass
class GoldenSmokeTestCase:
    """Represents a golden smoke test case."""
    name: str
    description: str
    repository_url: Optional[str] = None
    repository_path: Optional[str] = None
    local_files: Optional[Dict[str, str]] = None  # For synthetic repos
    budget: int = 10000
    tokenizer: str = "cl100k"
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)


class MockRepository:
    """Mock repository for golden smoke tests."""
    
    def __init__(self, name: str, files: Dict[str, str]):
        self.name = name
        self.files = files
    
    def create_on_disk(self, base_path: str) -> str:
        """Create repository structure on disk."""
        repo_path = os.path.join(base_path, self.name)
        os.makedirs(repo_path, exist_ok=True)
        
        for file_path, content in self.files.items():
            full_path = os.path.join(repo_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return repo_path


class GoldenSmokeTestSuite:
    """Main test suite for golden smoke flows."""
    
    def __init__(self):
        self.test_cases = self._define_golden_smoke_test_cases()
        self.oracle_registry = OracleRegistry()
        self.results = []
    
    def _define_golden_smoke_test_cases(self) -> List[GoldenSmokeTestCase]:
        """Define the standard golden smoke test cases."""
        return [
            # Small Python project
            GoldenSmokeTestCase(
                name="small_python_project",
                description="Small Python project with typical structure",
                local_files={
                    "main.py": '''#!/usr/bin/env python3
"""Main application module."""

import os
import sys
from typing import Dict, List, Optional

class DataProcessor:
    """Process data with various methods."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.processed_count = 0
    
    def process_file(self, filepath: str) -> Dict[str, Any]:
        """Process a single file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        result = {
            'filepath': filepath,
            'size': len(content),
            'lines': len(content.split('\\n')),
            'processed_at': time.time()
        }
        
        self.processed_count += 1
        return result
    
    def process_batch(self, filepaths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files."""
        results = []
        for filepath in filepaths:
            try:
                result = self.process_file(filepath)
                results.append(result)
            except Exception as e:
                results.append({
                    'filepath': filepath,
                    'error': str(e)
                })
        return results

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <file1> [file2] ...")
        sys.exit(1)
    
    processor = DataProcessor()
    files = sys.argv[1:]
    results = processor.process_batch(files)
    
    print(f"Processed {len(results)} files")
    for result in results:
        if 'error' in result:
            print(f"Error processing {result['filepath']}: {result['error']}")
        else:
            print(f"Processed {result['filepath']}: {result['size']} bytes, {result['lines']} lines")

if __name__ == "__main__":
    main()
''',
                    "utils.py": '''"""Utility functions for data processing."""

import json
import hashlib
from typing import Any, Dict, List, Union

def calculate_hash(data: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """Calculate hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data)
    return hash_obj.hexdigest()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

class ConfigValidator:
    """Validate configuration data."""
    
    REQUIRED_FIELDS = ['name', 'version', 'settings']
    
    @classmethod
    def validate(cls, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        for field in cls.REQUIRED_FIELDS:
            if field not in config:
                return False
        return True
    
    @classmethod
    def get_validation_errors(cls, config: Dict[str, Any]) -> List[str]:
        """Get list of validation errors."""
        errors = []
        for field in cls.REQUIRED_FIELDS:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        return errors
''',
                    "tests/test_main.py": '''"""Tests for main module."""

import unittest
import tempfile
import os
from unittest.mock import patch, mock_open
from main import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
    
    def test_process_file_success(self):
        """Test successful file processing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content\\nline 2")
            temp_path = f.name
        
        try:
            result = self.processor.process_file(temp_path)
            self.assertEqual(result['filepath'], temp_path)
            self.assertEqual(result['lines'], 2)
            self.assertIn('size', result)
            self.assertIn('processed_at', result)
        finally:
            os.unlink(temp_path)
    
    def test_process_file_not_found(self):
        """Test file not found error."""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_file('nonexistent_file.txt')
    
    def test_process_batch(self):
        """Test batch processing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1, \\
             tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
            
            f1.write("file 1 content")
            f2.write("file 2 content")
            temp_paths = [f1.name, f2.name]
        
        try:
            results = self.processor.process_batch(temp_paths)
            self.assertEqual(len(results), 2)
            self.assertNotIn('error', results[0])
            self.assertNotIn('error', results[1])
        finally:
            for path in temp_paths:
                os.unlink(path)
    
    def test_process_batch_with_errors(self):
        """Test batch processing with some errors."""
        paths = ['existing_file.txt', 'nonexistent_file.txt']
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            paths[0] = f.name
        
        try:
            results = self.processor.process_batch(paths)
            self.assertEqual(len(results), 2)
            self.assertNotIn('error', results[0])
            self.assertIn('error', results[1])
        finally:
            os.unlink(paths[0])

if __name__ == '__main__':
    unittest.main()
''',
                    "tests/test_utils.py": '''"""Tests for utils module."""

import unittest
import tempfile
import json
from utils import calculate_hash, ConfigValidator

class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_calculate_hash_string(self):
        """Test hash calculation for string."""
        result = calculate_hash("test string")
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)  # SHA256 hex length
    
    def test_calculate_hash_bytes(self):
        """Test hash calculation for bytes."""
        result = calculate_hash(b"test bytes")
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
    
    def test_calculate_hash_consistency(self):
        """Test hash calculation consistency."""
        data = "consistent test data"
        hash1 = calculate_hash(data)
        hash2 = calculate_hash(data)
        self.assertEqual(hash1, hash2)

class TestConfigValidator(unittest.TestCase):
    """Test configuration validator."""
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            'name': 'test_app',
            'version': '1.0.0',
            'settings': {'debug': True}
        }
        self.assertTrue(ConfigValidator.validate(config))
    
    def test_validate_invalid_config(self):
        """Test validation of invalid configuration."""
        config = {
            'name': 'test_app',
            'version': '1.0.0'
            # Missing 'settings'
        }
        self.assertFalse(ConfigValidator.validate(config))
    
    def test_get_validation_errors(self):
        """Test getting validation errors."""
        config = {'name': 'test_app'}  # Missing version and settings
        errors = ConfigValidator.get_validation_errors(config)
        self.assertEqual(len(errors), 2)
        self.assertIn('version', errors[0] + errors[1])
        self.assertIn('settings', errors[0] + errors[1])

if __name__ == '__main__':
    unittest.main()
''',
                    "config.json": '''{
  "name": "data_processor",
  "version": "1.0.0",
  "settings": {
    "debug": false,
    "max_file_size": 1048576,
    "supported_formats": ["txt", "json", "csv"],
    "output_format": "json"
  },
  "logging": {
    "level": "INFO",
    "file": "app.log"
  }
}''',
                    "README.md": '''# Data Processor

A simple data processing application written in Python.

## Features

- Process single files or batches of files
- Calculate file statistics (size, line count)
- Configurable processing options
- Comprehensive test suite

## Usage

```bash
python main.py file1.txt file2.txt
```

## Configuration

Edit `config.json` to customize processing options:

- `max_file_size`: Maximum file size to process (bytes)
- `supported_formats`: List of supported file extensions
- `debug`: Enable debug mode

## Testing

Run the test suite:

```bash
python -m unittest discover tests/
```

## Requirements

- Python 3.7+
- No external dependencies
''',
                    ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log

# OS
.DS_Store
Thumbs.db
'''
                },
                budget=8000,
                expected_outcomes={
                    'min_chunks': 5,
                    'max_chunks': 20,
                    'should_include_main': True,
                    'should_include_utils': True,
                    'should_include_tests': True
                },
                performance_targets={
                    'max_processing_time_ms': 5000,
                    'max_memory_usage_mb': 100
                }
            ),
            
            # JavaScript/TypeScript project
            GoldenSmokeTestCase(
                name="typescript_web_project", 
                description="TypeScript web application with React",
                local_files={
                    "src/App.tsx": '''import React, { useState, useEffect } from 'react';
import { UserService } from './services/UserService';
import { User } from './types/User';
import './App.css';

const App: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadUsers = async () => {
      try {
        setLoading(true);
        const userService = new UserService();
        const fetchedUsers = await userService.getAllUsers();
        setUsers(fetchedUsers);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    loadUsers();
  }, []);

  const handleUserUpdate = async (userId: string, updates: Partial<User>) => {
    try {
      const userService = new UserService();
      const updatedUser = await userService.updateUser(userId, updates);
      setUsers(prev => prev.map(user => 
        user.id === userId ? updatedUser : user
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Update failed');
    }
  };

  if (loading) return <div className="loading">Loading users...</div>;
  if (error) return <div className="error">Error: {error}</div>;

  return (
    <div className="App">
      <header className="App-header">
        <h1>User Management</h1>
      </header>
      <main>
        <div className="user-list">
          {users.map(user => (
            <div key={user.id} className="user-card">
              <h3>{user.name}</h3>
              <p>{user.email}</p>
              <button onClick={() => handleUserUpdate(user.id, { active: !user.active })}>
                {user.active ? 'Deactivate' : 'Activate'}
              </button>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
};

export default App;
''',
                    "src/services/UserService.ts": '''import { User } from '../types/User';
import { ApiClient } from '../utils/ApiClient';

export class UserService {
  private apiClient: ApiClient;

  constructor() {
    this.apiClient = new ApiClient('/api');
  }

  async getAllUsers(): Promise<User[]> {
    try {
      const response = await this.apiClient.get<User[]>('/users');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch users:', error);
      throw new Error('Failed to load users');
    }
  }

  async getUserById(id: string): Promise<User> {
    try {
      const response = await this.apiClient.get<User>(`/users/${id}`);
      return response.data;
    } catch (error) {
      console.error(`Failed to fetch user ${id}:`, error);
      throw new Error(`Failed to load user ${id}`);
    }
  }

  async createUser(userData: Omit<User, 'id' | 'createdAt'>): Promise<User> {
    try {
      const response = await this.apiClient.post<User>('/users', userData);
      return response.data;
    } catch (error) {
      console.error('Failed to create user:', error);
      throw new Error('Failed to create user');
    }
  }

  async updateUser(id: string, updates: Partial<User>): Promise<User> {
    try {
      const response = await this.apiClient.patch<User>(`/users/${id}`, updates);
      return response.data;
    } catch (error) {
      console.error(`Failed to update user ${id}:`, error);
      throw new Error(`Failed to update user ${id}`);
    }
  }

  async deleteUser(id: string): Promise<void> {
    try {
      await this.apiClient.delete(`/users/${id}`);
    } catch (error) {
      console.error(`Failed to delete user ${id}:`, error);
      throw new Error(`Failed to delete user ${id}`);
    }
  }
}
''',
                    "src/types/User.ts": '''export interface User {
  id: string;
  name: string;
  email: string;
  active: boolean;
  role: UserRole;
  createdAt: string;
  updatedAt?: string;
}

export enum UserRole {
  ADMIN = 'admin',
  USER = 'user',
  MODERATOR = 'moderator'
}

export interface CreateUserRequest {
  name: string;
  email: string;
  role: UserRole;
}

export interface UpdateUserRequest {
  name?: string;
  email?: string;
  active?: boolean;
  role?: UserRole;
}
''',
                    "src/utils/ApiClient.ts": '''export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export interface RequestConfig {
  headers?: Record<string, string>;
  timeout?: number;
}

export class ApiClient {
  private baseURL: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  private async request<T>(
    method: string,
    endpoint: string,
    data?: unknown,
    config?: RequestConfig
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    const headers = { ...this.defaultHeaders, ...config?.headers };

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: data ? JSON.stringify(data) : undefined,
        signal: config?.timeout ? AbortSignal.timeout(config.timeout) : undefined,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const responseData = await response.json();
      
      return {
        data: responseData,
        status: response.status,
      };
    } catch (error) {
      console.error(`API request failed: ${method} ${url}`, error);
      throw error;
    }
  }

  async get<T>(endpoint: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('GET', endpoint, undefined, config);
  }

  async post<T>(endpoint: string, data: unknown, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('POST', endpoint, data, config);
  }

  async patch<T>(endpoint: string, data: unknown, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('PATCH', endpoint, data, config);
  }

  async delete<T>(endpoint: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('DELETE', endpoint, undefined, config);
  }
}
''',
                    "package.json": '''{
  "name": "typescript-web-project",
  "version": "1.0.0",
  "description": "TypeScript web application with React",
  "main": "index.js",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint src --ext ts,tsx",
    "test": "jest",
    "test:watch": "jest --watch"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@typescript-eslint/eslint-plugin": "^5.57.0",
    "@typescript-eslint/parser": "^5.57.0",
    "@vitejs/plugin-react": "^4.0.0",
    "eslint": "^8.37.0",
    "eslint-plugin-react": "^7.32.0",
    "jest": "^29.5.0",
    "typescript": "^5.0.0",
    "vite": "^4.2.0"
  }
}''',
                    "tsconfig.json": '''{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}'''
                },
                budget=12000,
                expected_outcomes={
                    'min_chunks': 8,
                    'max_chunks': 25,
                    'should_include_components': True,
                    'should_include_services': True,
                    'should_include_types': True
                },
                performance_targets={
                    'max_processing_time_ms': 8000,
                    'max_memory_usage_mb': 150
                }
            ),
            
            # Documentation-heavy project
            GoldenSmokeTestCase(
                name="documentation_project",
                description="Project with extensive documentation and mixed file types",
                local_files={
                    "README.md": '''# Project Documentation

This is a comprehensive documentation project showcasing various documentation formats and structures.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [API Reference](#api-reference)
4. [Examples](#examples)
5. [Contributing](#contributing)

## Introduction

This project demonstrates best practices for documentation, including:

- Clear structure and navigation
- Code examples and snippets
- API documentation
- User guides and tutorials
- Development guidelines

## Getting Started

### Prerequisites

- Node.js 16+
- Python 3.8+
- Docker (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/example/docs-project.git
   cd docs-project
   ```

2. Install dependencies:
   ```bash
   npm install
   pip install -r requirements.txt
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## API Reference

See [API.md](./docs/API.md) for complete API documentation.

## Examples

Check the [examples/](./examples/) directory for usage examples.
''',
                    "docs/API.md": '''# API Reference

## Overview

This API provides endpoints for managing resources and data.

## Authentication

All API requests require authentication via Bearer token:

```
Authorization: Bearer <your-token>
```

## Endpoints

### Users

#### GET /api/users

Retrieve all users.

**Parameters:**
- `limit` (optional): Maximum number of users to return (default: 50)
- `offset` (optional): Number of users to skip (default: 0)
- `active` (optional): Filter by active status

**Response:**
```json
{
  "users": [
    {
      "id": "user123",
      "name": "John Doe",
      "email": "john@example.com",
      "active": true,
      "created_at": "2023-01-01T00:00:00Z"
    }
  ],
  "total": 1,
  "has_more": false
}
```

#### POST /api/users

Create a new user.

**Request Body:**
```json
{
  "name": "Jane Smith",
  "email": "jane@example.com",
  "role": "user"
}
```

**Response:**
```json
{
  "id": "user456",
  "name": "Jane Smith",
  "email": "jane@example.com",
  "role": "user",
  "active": true,
  "created_at": "2023-01-02T00:00:00Z"
}
```

### Projects

#### GET /api/projects

Retrieve all projects for the authenticated user.

**Response:**
```json
{
  "projects": [
    {
      "id": "proj123",
      "name": "My Project",
      "description": "A sample project",
      "owner_id": "user123",
      "created_at": "2023-01-01T00:00:00Z"
    }
  ]
}
```
''',
                    "docs/CONTRIBUTING.md": '''# Contributing Guidelines

Thank you for your interest in contributing to this project!

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

When creating a bug report, include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, version, etc.)
- Screenshots if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Use a clear, descriptive title
- Provide detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- Include examples if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation if needed
7. Submit a pull request

#### Pull Request Guidelines

- Use descriptive commit messages
- Keep changes focused and atomic
- Include tests for new features
- Update documentation as needed
- Follow existing code style
- Squash commits before merging

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/project.git
   cd project
   ```

2. Install dependencies:
   ```bash
   npm install
   pip install -r requirements-dev.txt
   ```

3. Run tests:
   ```bash
   npm test
   python -m pytest
   ```

## Code Style

### JavaScript/TypeScript
- Use ESLint and Prettier
- Follow TypeScript strict mode
- Use meaningful variable names
- Include JSDoc comments for public APIs

### Python
- Follow PEP 8
- Use type hints
- Include docstrings for all functions and classes
- Use meaningful variable names

### Documentation
- Use clear, concise language
- Include code examples
- Keep examples up to date
- Use proper Markdown formatting

## Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Test edge cases and error conditions
- Use descriptive test names

Thank you for contributing!
''',
                    "examples/basic_usage.py": '''#!/usr/bin/env python3
"""Basic usage examples for the project."""

import requests
import json
from typing import Dict, Any, List

class APIClient:
    """Simple API client for demonstration."""
    
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def get_users(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of users."""
        url = f"{self.base_url}/api/users"
        params = {'limit': limit}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        return response.json()['users']
    
    def create_user(self, name: str, email: str, role: str = 'user') -> Dict[str, Any]:
        """Create a new user."""
        url = f"{self.base_url}/api/users"
        data = {
            'name': name,
            'email': email,
            'role': role
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()

def main():
    """Demonstrate basic API usage."""
    # Initialize client
    client = APIClient('https://api.example.com', 'your-token-here')
    
    # Get existing users
    print("Fetching users...")
    users = client.get_users(limit=10)
    print(f"Found {len(users)} users")
    
    for user in users:
        print(f"- {user['name']} ({user['email']})")
    
    # Create a new user
    print("\\nCreating new user...")
    new_user = client.create_user(
        name="Example User",
        email="example@test.com",
        role="user"
    )
    
    print(f"Created user: {new_user['name']} with ID {new_user['id']}")

if __name__ == "__main__":
    main()
''',
                    "examples/advanced_usage.js": '''/**
 * Advanced usage examples for the JavaScript/TypeScript client.
 */

class AdvancedAPIClient {
  constructor(baseUrl, token) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.token = token;
    this.defaultHeaders = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  }

  async request(method, endpoint, data = null, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = { ...this.defaultHeaders, ...options.headers };
    
    const config = {
      method,
      headers,
      ...options
    };
    
    if (data && ['POST', 'PUT', 'PATCH'].includes(method)) {
      config.body = JSON.stringify(data);
    }
    
    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${method} ${url}`, error);
      throw error;
    }
  }

  async getUsersWithPagination(limit = 50, offset = 0) {
    const params = new URLSearchParams({ limit, offset });
    return this.request('GET', `/api/users?${params}`);
  }

  async batchCreateUsers(users) {
    const promises = users.map(user => 
      this.request('POST', '/api/users', user)
    );
    
    try {
      return await Promise.all(promises);
    } catch (error) {
      console.error('Batch user creation failed:', error);
      throw error;
    }
  }

  async updateUserWithRetry(userId, updates, maxRetries = 3) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await this.request('PATCH', `/api/users/${userId}`, updates);
      } catch (error) {
        lastError = error;
        
        if (attempt < maxRetries) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw new Error(`Failed after ${maxRetries} attempts: ${lastError.message}`);
  }

  async streamUsers(callback, batchSize = 100) {
    let offset = 0;
    let hasMore = true;
    
    while (hasMore) {
      const response = await this.getUsersWithPagination(batchSize, offset);
      const users = response.users || [];
      
      if (users.length === 0) {
        hasMore = false;
      } else {
        await callback(users);
        offset += users.length;
        hasMore = response.has_more || users.length === batchSize;
      }
    }
  }
}

// Usage examples
async function demonstrateAdvancedFeatures() {
  const client = new AdvancedAPIClient('https://api.example.com', 'your-token');
  
  // Example 1: Batch operations
  console.log('Creating multiple users...');
  const usersToCreate = [
    { name: 'Alice Johnson', email: 'alice@example.com', role: 'user' },
    { name: 'Bob Smith', email: 'bob@example.com', role: 'moderator' },
    { name: 'Carol Wilson', email: 'carol@example.com', role: 'user' }
  ];
  
  try {
    const createdUsers = await client.batchCreateUsers(usersToCreate);
    console.log(`Successfully created ${createdUsers.length} users`);
  } catch (error) {
    console.error('Batch creation failed:', error.message);
  }
  
  // Example 2: Streaming large datasets
  console.log('\\nStreaming all users...');
  let totalUsers = 0;
  
  await client.streamUsers(async (userBatch) => {
    console.log(`Processing batch of ${userBatch.length} users`);
    totalUsers += userBatch.length;
    
    // Process each user in the batch
    for (const user of userBatch) {
      // Simulate some processing
      if (!user.active) {
        console.log(`Inactive user found: ${user.name}`);
      }
    }
  });
  
  console.log(`Total users processed: ${totalUsers}`);
  
  // Example 3: Error handling with retries
  console.log('\\nUpdating user with retry logic...');
  try {
    const updatedUser = await client.updateUserWithRetry('user123', {
      name: 'Updated Name',
      active: true
    });
    console.log('User updated successfully:', updatedUser.name);
  } catch (error) {
    console.error('Failed to update user:', error.message);
  }
}

// Run examples
demonstrateAdvancedFeatures().catch(console.error);
''',
                    "requirements.txt": '''requests>=2.28.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=22.0.0
flake8>=5.0.0
mypy>=1.0.0
'''
                },
                budget=15000,
                expected_outcomes={
                    'min_chunks': 10,
                    'max_chunks': 30,
                    'should_include_markdown': True,
                    'should_include_code_examples': True,
                    'should_include_config_files': True
                },
                performance_targets={
                    'max_processing_time_ms': 10000,
                    'max_memory_usage_mb': 200
                }
            )
        ]
    
    def run_golden_smoke_test(self, test_case: GoldenSmokeTestCase, temp_dir: str) -> Dict[str, Any]:
        """Run a single golden smoke test case."""
        print(f"Running golden smoke test: {test_case.name}")
        
        # Create repository on disk
        if test_case.local_files:
            mock_repo = MockRepository(test_case.name, test_case.local_files)
            repo_path = mock_repo.create_on_disk(temp_dir)
        else:
            # Would handle real repository cloning here
            raise NotImplementedError("Real repository cloning not implemented in this test")
        
        start_time = time.time()
        
        try:
            # Import PackRepo components (mock for now)
            from packrepo.packer.selector.selector import Selector
            from packrepo.packer.chunker.chunker import CodeChunker
            from packrepo.packer.tokenizer.implementations import get_tokenizer
            from packrepo.packer.packfmt.base import PackIndex, PackSection
            
            # Initialize components
            chunker = CodeChunker()
            selector = Selector()
            tokenizer = get_tokenizer(test_case.tokenizer)
            
            # Process repository files
            all_chunks = []
            total_tokens = 0
            
            for file_path, content in test_case.local_files.items():
                try:
                    file_chunks = chunker.chunk_file(content, file_path)
                    
                    for chunk in file_chunks:
                        chunk_tokens = tokenizer.count_tokens(chunk.content)
                        chunk_data = {
                            'id': f'chunk_{len(all_chunks)}',
                            'rel_path': file_path,
                            'start_line': chunk.start_line,
                            'end_line': chunk.end_line,
                            'content': chunk.content,
                            'tokens': chunk_tokens,
                            'cost': chunk_tokens,
                            'score': random.uniform(0.1, 1.0)  # Mock scoring
                        }
                        all_chunks.append(chunk_data)
                        total_tokens += chunk_tokens
                        
                except Exception as e:
                    print(f"Warning: Failed to chunk file {file_path}: {e}")
                    continue
            
            # Run selection
            selected_chunks = selector.select(all_chunks, test_case.budget)
            
            # Calculate metrics
            selected_tokens = sum(chunk.get('tokens', 0) for chunk in selected_chunks)
            processing_time = (time.time() - start_time) * 1000  # milliseconds
            
            # Create pack format
            pack_sections = []
            pack_chunks = []
            
            for chunk in selected_chunks:
                section = PackSection(
                    rel_path=chunk['rel_path'],
                    start_line=chunk['start_line'],
                    end_line=chunk['end_line'],
                    content=chunk['content'],
                    mode='full'
                )
                pack_sections.append(section)
                pack_chunks.append({
                    'id': chunk['id'],
                    'rel_path': chunk['rel_path'],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line'],
                    'selected_tokens': chunk['tokens']
                })
            
            pack_index = PackIndex(
                target_budget=test_case.budget,
                actual_tokens=selected_tokens,
                chunks=pack_chunks,
                tokenizer_name=test_case.tokenizer,
                tokenizer_version="1.0.0"
            )
            
            pack = PackFormat(index=pack_index, sections=pack_sections)
            
            # Run oracle validations
            oracle_results = []
            for oracle_name, oracle in self.oracle_registry.get_all_oracles().items():
                try:
                    result = oracle.validate(pack)
                    oracle_results.append({
                        'oracle': oracle_name,
                        'result': result.result.value,
                        'message': result.message
                    })
                except Exception as e:
                    oracle_results.append({
                        'oracle': oracle_name,
                        'result': 'ERROR',
                        'message': str(e)
                    })
            
            # Validate expected outcomes
            outcome_validations = self._validate_expected_outcomes(
                test_case.expected_outcomes,
                selected_chunks,
                pack
            )
            
            # Check performance targets
            performance_validations = self._validate_performance_targets(
                test_case.performance_targets,
                processing_time,
                selected_tokens
            )
            
            return {
                'test_case': test_case.name,
                'status': 'SUCCESS',
                'processing_time_ms': processing_time,
                'total_chunks_generated': len(all_chunks),
                'selected_chunks': len(selected_chunks),
                'budget_utilization': selected_tokens / test_case.budget,
                'oracle_results': oracle_results,
                'outcome_validations': outcome_validations,
                'performance_validations': performance_validations,
                'pack_hash': hashlib.sha256(str(pack).encode()).hexdigest()[:16]
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return {
                'test_case': test_case.name,
                'status': 'FAILED',
                'error': str(e),
                'processing_time_ms': processing_time
            }
    
    def _validate_expected_outcomes(self, expected: Dict[str, Any], 
                                   selected_chunks: List[Dict[str, Any]], 
                                   pack: PackFormat) -> Dict[str, bool]:
        """Validate expected outcomes against actual results."""
        validations = {}
        
        # Check chunk count bounds
        if 'min_chunks' in expected:
            validations['min_chunks'] = len(selected_chunks) >= expected['min_chunks']
        
        if 'max_chunks' in expected:
            validations['max_chunks'] = len(selected_chunks) <= expected['max_chunks']
        
        # Check file inclusion requirements
        selected_files = set(chunk['rel_path'] for chunk in selected_chunks)
        
        if expected.get('should_include_main'):
            validations['includes_main'] = any('main.py' in f or 'main.' in f for f in selected_files)
        
        if expected.get('should_include_utils'):
            validations['includes_utils'] = any('utils' in f for f in selected_files)
        
        if expected.get('should_include_tests'):
            validations['includes_tests'] = any('test' in f for f in selected_files)
        
        if expected.get('should_include_components'):
            validations['includes_components'] = any('component' in f.lower() or 'App.' in f for f in selected_files)
        
        if expected.get('should_include_services'):
            validations['includes_services'] = any('service' in f.lower() for f in selected_files)
        
        if expected.get('should_include_types'):
            validations['includes_types'] = any('type' in f.lower() for f in selected_files)
        
        if expected.get('should_include_markdown'):
            validations['includes_markdown'] = any(f.endswith('.md') for f in selected_files)
        
        if expected.get('should_include_code_examples'):
            validations['includes_code_examples'] = any('example' in f.lower() for f in selected_files)
        
        if expected.get('should_include_config_files'):
            validations['includes_config_files'] = any(f.endswith('.json') or 'config' in f for f in selected_files)
        
        return validations
    
    def _validate_performance_targets(self, targets: Dict[str, float], 
                                    processing_time_ms: float,
                                    selected_tokens: int) -> Dict[str, bool]:
        """Validate performance targets against actual metrics."""
        validations = {}
        
        if 'max_processing_time_ms' in targets:
            validations['processing_time'] = processing_time_ms <= targets['max_processing_time_ms']
        
        # Memory usage validation would need process monitoring
        if 'max_memory_usage_mb' in targets:
            validations['memory_usage'] = True  # Placeholder - would need real memory monitoring
        
        return validations
    
    def run_all_golden_smoke_tests(self) -> List[Dict[str, Any]]:
        """Run all golden smoke test cases."""
        results = []
        
        with tempfile.TemporaryDirectory(prefix="golden_smoke_") as temp_dir:
            for test_case in self.test_cases:
                result = self.run_golden_smoke_test(test_case, temp_dir)
                results.append(result)
                self.results.append(result)
        
        return results
    
    def generate_smoke_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive smoke test report."""
        if not self.results:
            return {'error': 'No test results available'}
        
        successful_tests = [r for r in self.results if r['status'] == 'SUCCESS']
        failed_tests = [r for r in self.results if r['status'] == 'FAILED']
        
        # Calculate aggregate metrics
        total_tests = len(self.results)
        success_rate = len(successful_tests) / total_tests if total_tests > 0 else 0
        
        avg_processing_time = sum(r['processing_time_ms'] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_budget_utilization = sum(r['budget_utilization'] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        
        # Oracle results summary
        oracle_summary = {}
        for result in successful_tests:
            for oracle_result in result.get('oracle_results', []):
                oracle_name = oracle_result['oracle']
                if oracle_name not in oracle_summary:
                    oracle_summary[oracle_name] = {'pass': 0, 'fail': 0, 'error': 0}
                
                if oracle_result['result'] == 'PASS':
                    oracle_summary[oracle_name]['pass'] += 1
                elif oracle_result['result'] == 'FAIL':
                    oracle_summary[oracle_name]['fail'] += 1
                else:
                    oracle_summary[oracle_name]['error'] += 1
        
        return {
            'total_tests': total_tests,
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': success_rate,
            'avg_processing_time_ms': avg_processing_time,
            'avg_budget_utilization': avg_budget_utilization,
            'oracle_summary': oracle_summary,
            'individual_results': self.results
        }


class TestGoldenSmokeFlows:
    """Test class for golden smoke flows."""
    
    @pytest.fixture
    def smoke_test_suite(self):
        """Create golden smoke test suite."""
        return GoldenSmokeTestSuite()
    
    def test_small_python_project_smoke(self, smoke_test_suite):
        """Test small Python project golden smoke flow."""
        test_case = next(tc for tc in smoke_test_suite.test_cases if tc.name == "small_python_project")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = smoke_test_suite.run_golden_smoke_test(test_case, temp_dir)
        
        # Validate smoke test results
        assert result['status'] == 'SUCCESS', f"Smoke test failed: {result.get('error', 'Unknown error')}"
        
        # Check basic metrics
        assert result['selected_chunks'] > 0, "Should select some chunks"
        assert 0.5 <= result['budget_utilization'] <= 1.0, f"Budget utilization should be reasonable: {result['budget_utilization']}"
        
        # Check performance
        assert result['processing_time_ms'] < 10000, f"Processing time too high: {result['processing_time_ms']}ms"
        
        # Validate expected outcomes
        outcome_validations = result['outcome_validations']
        
        if 'min_chunks' in outcome_validations:
            assert outcome_validations['min_chunks'], "Should meet minimum chunk count"
        
        if 'includes_main' in outcome_validations:
            assert outcome_validations['includes_main'], "Should include main.py"
        
        # Check oracle results
        oracle_results = result['oracle_results']
        oracle_passes = sum(1 for or_result in oracle_results if or_result['result'] == 'PASS')
        oracle_total = len(oracle_results)
        
        if oracle_total > 0:
            oracle_pass_rate = oracle_passes / oracle_total
            assert oracle_pass_rate > 0.8, f"Oracle pass rate too low: {oracle_pass_rate:.2%}"
    
    def test_typescript_project_smoke(self, smoke_test_suite):
        """Test TypeScript project golden smoke flow."""
        test_case = next(tc for tc in smoke_test_suite.test_cases if tc.name == "typescript_web_project")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = smoke_test_suite.run_golden_smoke_test(test_case, temp_dir)
        
        assert result['status'] == 'SUCCESS', f"TypeScript smoke test failed: {result.get('error', 'Unknown error')}"
        
        # TypeScript-specific validations
        outcome_validations = result['outcome_validations']
        
        if 'includes_components' in outcome_validations:
            assert outcome_validations['includes_components'], "Should include React components"
        
        if 'includes_services' in outcome_validations:
            assert outcome_validations['includes_services'], "Should include service classes"
        
        if 'includes_types' in outcome_validations:
            assert outcome_validations['includes_types'], "Should include TypeScript types"
    
    def test_documentation_project_smoke(self, smoke_test_suite):
        """Test documentation project golden smoke flow."""
        test_case = next(tc for tc in smoke_test_suite.test_cases if tc.name == "documentation_project")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = smoke_test_suite.run_golden_smoke_test(test_case, temp_dir)
        
        assert result['status'] == 'SUCCESS', f"Documentation smoke test failed: {result.get('error', 'Unknown error')}"
        
        # Documentation-specific validations  
        outcome_validations = result['outcome_validations']
        
        if 'includes_markdown' in outcome_validations:
            assert outcome_validations['includes_markdown'], "Should include Markdown documentation"
        
        if 'includes_code_examples' in outcome_validations:
            assert outcome_validations['includes_code_examples'], "Should include code examples"
    
    def test_comprehensive_smoke_suite(self, smoke_test_suite):
        """Test complete golden smoke test suite."""
        results = smoke_test_suite.run_all_golden_smoke_tests()
        
        # Validate overall results
        assert len(results) > 0, "Should run some smoke tests"
        
        successful_results = [r for r in results if r['status'] == 'SUCCESS']
        overall_success_rate = len(successful_results) / len(results)
        
        print(f"Golden smoke test suite results:")
        print(f"  Total tests: {len(results)}")
        print(f"  Successful: {len(successful_results)}")
        print(f"  Success rate: {overall_success_rate:.2%}")
        
        # Show individual results
        for result in results:
            status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"  {status_icon} {result['test_case']}: {result['processing_time_ms']:.0f}ms")
            
            if result['status'] == 'FAILED':
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        # Overall success rate should be high
        assert overall_success_rate > 0.8, f"Golden smoke test success rate insufficient: {overall_success_rate:.2%}"
        
        # Generate and validate report
        report = smoke_test_suite.generate_smoke_test_report()
        
        assert report['success_rate'] > 0.8, "Report should show high success rate"
        assert report['avg_processing_time_ms'] < 20000, "Average processing time should be reasonable"
        assert 0.5 <= report['avg_budget_utilization'] <= 1.0, "Average budget utilization should be reasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])