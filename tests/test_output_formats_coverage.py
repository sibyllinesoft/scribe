#!/usr/bin/env python3
"""
Targeted tests for output_formats module to increase coverage.
Focus on covering the uncovered lines 108-135 and 140-152.
"""

import pytest
from unittest.mock import patch, MagicMock

from scribe.output_formats import _detect_language, _highlight_code


class TestDetectLanguage:
    """Test the _detect_language function for coverage of lines 108-135."""
    
    def test_detect_language_python(self):
        """Test Python file detection."""
        assert _detect_language("script.py") == "python"
    
    def test_detect_language_javascript_variants(self):
        """Test JavaScript and variants detection."""
        assert _detect_language("app.js") == "javascript"
        assert _detect_language("component.jsx") == "javascript"  
        assert _detect_language("module.mjs") == "javascript"
        assert _detect_language("config.cjs") == "javascript"
    
    def test_detect_language_typescript_variants(self):
        """Test TypeScript variants detection."""
        assert _detect_language("app.ts") == "typescript"
        assert _detect_language("component.tsx") == "typescript"
    
    def test_detect_language_html_variants(self):
        """Test HTML variants detection."""
        assert _detect_language("index.html") == "html"
        assert _detect_language("old.htm") == "html"
        assert _detect_language("doc.xhtml") == "html"
    
    def test_detect_language_css_variants(self):
        """Test CSS and preprocessors detection."""
        assert _detect_language("style.css") == "css"
        assert _detect_language("style.scss") == "scss"
        assert _detect_language("style.sass") == "sass"
        assert _detect_language("style.less") == "less"
    
    def test_detect_language_json_variants(self):
        """Test JSON variants detection."""
        assert _detect_language("config.json") == "json"
        assert _detect_language("config.jsonc") == "json"
        assert _detect_language("config.json5") == "json"
    
    def test_detect_language_xml_variants(self):
        """Test XML variants detection."""
        assert _detect_language("config.xml") == "xml"
        assert _detect_language("transform.xsl") == "xml"
        assert _detect_language("schema.xsd") == "xml"
    
    def test_detect_language_yaml_variants(self):
        """Test YAML variants detection."""
        assert _detect_language("config.yml") == "yaml"
        assert _detect_language("config.yaml") == "yaml"
    
    def test_detect_language_markdown_variants(self):
        """Test Markdown variants detection."""
        assert _detect_language("README.md") == "markdown"
        assert _detect_language("doc.markdown") == "markdown"
        assert _detect_language("guide.mdown") == "markdown"
    
    def test_detect_language_shell_variants(self):
        """Test Shell script variants detection."""
        assert _detect_language("script.sh") == "bash"
        assert _detect_language("script.bash") == "bash"
        assert _detect_language("config.zsh") == "zsh"
        assert _detect_language("config.fish") == "fish"
    
    def test_detect_language_rust(self):
        """Test Rust detection."""
        assert _detect_language("main.rs") == "rust"
    
    def test_detect_language_go(self):
        """Test Go detection."""
        assert _detect_language("main.go") == "go"
    
    def test_detect_language_jvm_languages(self):
        """Test JVM languages detection."""
        assert _detect_language("Main.java") == "java"
        assert _detect_language("App.kt") == "kotlin"
        assert _detect_language("App.scala") == "scala"
    
    def test_detect_language_c_cpp_variants(self):
        """Test C/C++ variants detection."""
        assert _detect_language("program.c") == "c"
        assert _detect_language("program.cpp") == "cpp"
        assert _detect_language("program.cc") == "cpp"
        assert _detect_language("program.cxx") == "cpp"
        assert _detect_language("header.h") == "c"
        assert _detect_language("header.hpp") == "cpp"
        assert _detect_language("header.hxx") == "cpp"
    
    def test_detect_language_other_languages(self):
        """Test other language detections."""
        assert _detect_language("Program.cs") == "csharp"
        assert _detect_language("script.php") == "php"
        assert _detect_language("script.rb") == "ruby"
        assert _detect_language("query.sql") == "sql"
        assert _detect_language("analysis.r") == "r"
        assert _detect_language("App.swift") == "swift"
        assert _detect_language("app.dart") == "dart"
    
    def test_detect_language_unknown_extension(self):
        """Test unknown extension defaults to 'text'."""
        assert _detect_language("file.unknown") == "text"
        assert _detect_language("file.xyz") == "text"
        assert _detect_language("file") == "text"  # No extension
    
    def test_detect_language_case_insensitive(self):
        """Test that detection is case insensitive."""
        assert _detect_language("FILE.PY") == "python"
        assert _detect_language("Script.JS") == "javascript"
        assert _detect_language("Config.JSON") == "json"


class TestHighlightCode:
    """Test the _highlight_code function for coverage of lines 140-152."""
    
    @patch('scribe.output_formats.get_lexer_for_filename')
    @patch('scribe.output_formats.highlight')
    def test_highlight_code_success(self, mock_highlight, mock_get_lexer):
        """Test successful code highlighting."""
        mock_lexer = MagicMock()
        mock_get_lexer.return_value = mock_lexer
        mock_highlight.return_value = "<highlighted code>"
        
        result = _highlight_code("print('hello')", "test.py")
        
        assert result == "<highlighted code>"
        mock_get_lexer.assert_called_once_with("test.py", stripall=False)
        mock_highlight.assert_called_once()
        
        # Just check that the function completes successfully
        # The formatter configuration is tested by execution
    
    @patch('scribe.output_formats.get_lexer_for_filename')
    @patch('scribe.output_formats.TextLexer')
    @patch('scribe.output_formats.highlight')
    def test_highlight_code_lexer_exception_fallback(self, mock_highlight, mock_text_lexer, mock_get_lexer):
        """Test fallback to TextLexer when get_lexer_for_filename fails."""
        mock_get_lexer.side_effect = Exception("Lexer not found")
        mock_text_lexer_instance = MagicMock()
        mock_text_lexer.return_value = mock_text_lexer_instance
        mock_highlight.return_value = "<highlighted as text>"
        
        result = _highlight_code("some code", "unknown.extension")
        
        assert result == "<highlighted as text>"
        mock_get_lexer.assert_called_once_with("unknown.extension", stripall=False)
        mock_text_lexer.assert_called_once_with(stripall=False)
        mock_highlight.assert_called_once_with("some code", mock_text_lexer_instance, mock_highlight.call_args[0][2])
    
    @patch('scribe.output_formats.get_lexer_for_filename')
    @patch('scribe.output_formats.highlight')
    def test_highlight_code_various_filenames(self, mock_highlight, mock_get_lexer):
        """Test highlighting with various filename types."""
        mock_lexer = MagicMock()
        mock_get_lexer.return_value = mock_lexer
        mock_highlight.return_value = "<highlighted>"
        
        filenames = ["script.py", "app.js", "style.css", "config.json", "README.md"]
        
        for filename in filenames:
            result = _highlight_code("code content", filename)
            assert result == "<highlighted>"
        
        assert mock_get_lexer.call_count == 5
        assert mock_highlight.call_count == 5
    
    @patch('scribe.output_formats.get_lexer_for_filename')
    @patch('scribe.output_formats.highlight')  
    def test_highlight_code_empty_content(self, mock_highlight, mock_get_lexer):
        """Test highlighting with empty content."""
        mock_lexer = MagicMock()
        mock_get_lexer.return_value = mock_lexer
        mock_highlight.return_value = "<empty highlighted>"
        
        result = _highlight_code("", "test.py")
        
        assert result == "<empty highlighted>"
        mock_highlight.assert_called_once_with("", mock_lexer, mock_highlight.call_args[0][2])
    
    @patch('scribe.output_formats.get_lexer_for_filename')
    @patch('scribe.output_formats.highlight')
    def test_highlight_code_multiline_content(self, mock_highlight, mock_get_lexer):
        """Test highlighting with multiline content."""
        mock_lexer = MagicMock()
        mock_get_lexer.return_value = mock_lexer  
        mock_highlight.return_value = "<multiline highlighted>"
        
        multiline_code = """def hello():
    print("Hello, World!")
    return True"""
        
        result = _highlight_code(multiline_code, "test.py")
        
        assert result == "<multiline highlighted>"
        mock_highlight.assert_called_once_with(multiline_code, mock_lexer, mock_highlight.call_args[0][2])