//! Code-aware tokenization for better search relevance
//!
//! This tokenizer handles:
//! - CamelCase splitting (RedirectTrailingSlash -> redirect, trailing, slash)
//! - snake_case splitting (redirect_trailing_slash -> redirect, trailing, slash)
//! - Preserves original tokens alongside splits
//! - Handles common programming constructs

use tantivy::tokenizer::{
    BoxTokenStream, Token, TokenStream, Tokenizer, TextAnalyzer, LowerCaser,
    SimpleTokenizer, RemoveLongFilter,
};

/// Code-aware tokenizer that splits identifiers
#[derive(Clone)]
pub struct CodeTokenizer;

impl Tokenizer for CodeTokenizer {
    type TokenStream<'a> = CodeTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        CodeTokenStream::new(text)
    }
}

/// Token stream that handles code identifier splitting
pub struct CodeTokenStream<'a> {
    text: &'a str,
    tokens: Vec<Token>,
    index: usize,
}

impl<'a> CodeTokenStream<'a> {
    fn new(text: &'a str) -> Self {
        let tokens = tokenize_code(text);
        Self {
            text,
            tokens,
            index: 0,
        }
    }
}

impl<'a> TokenStream for CodeTokenStream<'a> {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

/// Tokenize code text, splitting identifiers
fn tokenize_code(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut offset = 0;

    for word in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.is_empty() {
            continue;
        }

        let word_start = text[offset..].find(word).map(|i| offset + i).unwrap_or(offset);
        let word_end = word_start + word.len();

        // Add the original token
        tokens.push(Token {
            offset_from: word_start,
            offset_to: word_end,
            position: tokens.len(),
            text: word.to_lowercase(),
            position_length: 1,
        });

        // Split camelCase and PascalCase
        let parts = split_identifier(word);
        if parts.len() > 1 {
            for part in parts {
                if !part.is_empty() {
                    tokens.push(Token {
                        offset_from: word_start,
                        offset_to: word_end,
                        position: tokens.len(),
                        text: part.to_lowercase(),
                        position_length: 1,
                    });
                }
            }
        }

        offset = word_end;
    }

    tokens
}

/// Split an identifier by camelCase and snake_case boundaries
fn split_identifier(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut last = 0;

    // First split by underscore
    for segment in s.split('_') {
        if segment.is_empty() {
            continue;
        }

        // Then split by camelCase
        let chars: Vec<char> = segment.chars().collect();
        let mut part_start = 0;

        for i in 1..chars.len() {
            let prev = chars[i - 1];
            let curr = chars[i];

            // Split on lowercase->uppercase transition
            if prev.is_lowercase() && curr.is_uppercase() {
                if i > part_start {
                    let start = chars[..part_start].iter().collect::<String>().len();
                    let end = chars[..i].iter().collect::<String>().len();
                    if let Some(part) = segment.get(start..end) {
                        parts.push(part);
                    }
                }
                part_start = i;
            }
            // Split on uppercase->uppercase followed by lowercase (e.g., XMLParser -> XML, Parser)
            else if i + 1 < chars.len()
                && prev.is_uppercase()
                && curr.is_uppercase()
                && chars[i + 1].is_lowercase()
            {
                if i > part_start {
                    let start = chars[..part_start].iter().collect::<String>().len();
                    let end = chars[..i].iter().collect::<String>().len();
                    if let Some(part) = segment.get(start..end) {
                        parts.push(part);
                    }
                }
                part_start = i;
            }
        }

        // Add remaining part
        if part_start < chars.len() {
            let start = chars[..part_start].iter().collect::<String>().len();
            if let Some(part) = segment.get(start..) {
                parts.push(part);
            }
        }
    }

    if parts.is_empty() {
        parts.push(s);
    }

    parts
}

/// Register the code tokenizer with tantivy
pub fn register_code_tokenizer(index: &tantivy::Index) {
    let tokenizer = TextAnalyzer::builder(CodeTokenizer)
        .filter(RemoveLongFilter::limit(100))
        .filter(LowerCaser)
        .build();

    index.tokenizers().register("code", tokenizer);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_camel_case() {
        let parts = split_identifier("RedirectTrailingSlash");
        assert_eq!(parts, vec!["Redirect", "Trailing", "Slash"]);
    }

    #[test]
    fn test_split_snake_case() {
        let parts = split_identifier("redirect_trailing_slash");
        assert_eq!(parts, vec!["redirect", "trailing", "slash"]);
    }

    #[test]
    fn test_split_mixed() {
        let parts = split_identifier("XMLParser");
        assert_eq!(parts, vec!["XML", "Parser"]);
    }

    #[test]
    fn test_tokenize_code() {
        let tokens = tokenize_code("func RedirectTrailingSlash()");
        let texts: Vec<_> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"redirect"));
        assert!(texts.contains(&"trailing"));
        assert!(texts.contains(&"slash"));
    }
}
