//! XML output formatting for covering set results

use std::collections::HashMap;
use std::io::Write;

/// Escape special XML characters
pub fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Output covering set result as XML to stdout (for agent consumption)
pub fn output_covering_set_xml(
    result: &scribe_selection::CoveringSetResult,
    file_contents: &HashMap<String, String>,
    granularity: scribe_selection::CoveringSetGranularity,
) -> Result<(), Box<dyn std::error::Error>> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();

    writeln!(handle, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(handle, "<covering_set>")?;

    write_xml_target(&mut handle, &result.target_entity)?;

    if granularity == scribe_selection::CoveringSetGranularity::Entity {
        write_xml_entities(&mut handle, &result.entities)?;
    } else {
        write_xml_files(&mut handle, &result.files, file_contents)?;
    }

    write_xml_statistics(&mut handle, &result.statistics)?;
    writeln!(handle, "</covering_set>")?;

    Ok(())
}

/// Write XML target element
fn write_xml_target<W: Write>(
    handle: &mut W,
    target: &Option<scribe_selection::EntityLocation>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(target) = target {
        writeln!(handle, "  <target>")?;
        writeln!(handle, "    <file>{}</file>", escape_xml(&target.file_path))?;
        writeln!(
            handle,
            "    <name>{}</name>",
            escape_xml(&target.entity_name)
        )?;
        writeln!(
            handle,
            "    <type>{}</type>",
            escape_xml(&target.entity_type)
        )?;
        writeln!(
            handle,
            "    <lines start=\"{}\" end=\"{}\"/>",
            target.start_line, target.end_line
        )?;
        writeln!(handle, "  </target>")?;
    }
    Ok(())
}

/// Format a single entity as XML string
fn format_xml_entity(entity: &scribe_selection::CoveringSetEntity) -> String {
    format!(
        "    <entity>\n\
         \x20     <file>{}</file>\n\
         \x20     <name>{}</name>\n\
         \x20     <type>{}</type>\n\
         \x20     <lines start=\"{}\" end=\"{}\"/>\n\
         \x20     <distance>{}</distance>\n\
         \x20     <reason>{:?}</reason>\n\
         \x20     <content><![CDATA[{}]]></content>\n\
         \x20   </entity>",
        escape_xml(&entity.file_path),
        escape_xml(&entity.name),
        escape_xml(&entity.entity_type),
        entity.start_line,
        entity.end_line,
        entity.distance,
        entity.reason,
        entity.content
    )
}

/// Write XML entities element
fn write_xml_entities<W: Write>(
    handle: &mut W,
    entities: &[scribe_selection::CoveringSetEntity],
) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(handle, "  <entities count=\"{}\">", entities.len())?;
    for entity in entities {
        writeln!(handle, "{}", format_xml_entity(entity))?;
    }
    writeln!(handle, "  </entities>")?;
    Ok(())
}

/// Format a single file as XML string
fn format_xml_file(file: &scribe_selection::CoveringSetFile, content: &str) -> String {
    format!(
        "    <file>\n\
         \x20     <path>{}</path>\n\
         \x20     <distance>{}</distance>\n\
         \x20     <reason>{:?}</reason>\n\
         \x20     <content><![CDATA[{}]]></content>\n\
         \x20   </file>",
        escape_xml(&file.path),
        file.distance,
        file.reason,
        content
    )
}

/// Write XML files element
fn write_xml_files<W: Write>(
    handle: &mut W,
    files: &[scribe_selection::CoveringSetFile],
    file_contents: &HashMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(handle, "  <files count=\"{}\">", files.len())?;
    for file in files {
        let content = file_contents
            .get(&file.path)
            .map(|s| s.as_str())
            .unwrap_or("");
        writeln!(handle, "{}", format_xml_file(file, content))?;
    }
    writeln!(handle, "  </files>")?;
    Ok(())
}

/// Write XML statistics element
fn write_xml_statistics<W: Write>(
    handle: &mut W,
    stats: &scribe_selection::CoveringSetStatistics,
) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(handle, "  <statistics>")?;
    writeln!(
        handle,
        "    <files_examined>{}</files_examined>",
        stats.files_examined
    )?;
    writeln!(
        handle,
        "    <files_selected>{}</files_selected>",
        stats.files_selected
    )?;
    writeln!(
        handle,
        "    <entities_selected>{}</entities_selected>",
        stats.entities_selected
    )?;
    writeln!(
        handle,
        "    <max_depth_reached>{}</max_depth_reached>",
        stats.max_depth_reached
    )?;
    writeln!(
        handle,
        "    <limits_reached>{}</limits_reached>",
        stats.limits_reached
    )?;
    writeln!(handle, "  </statistics>")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scribe_selection::{
        CoveringSetEntity, CoveringSetFile, CoveringSetStatistics, EntityLocation, InclusionReason,
    };

    #[test]
    fn test_escape_xml_ampersand() {
        assert_eq!(escape_xml("foo & bar"), "foo &amp; bar");
    }

    #[test]
    fn test_escape_xml_less_than() {
        assert_eq!(escape_xml("a < b"), "a &lt; b");
    }

    #[test]
    fn test_escape_xml_greater_than() {
        assert_eq!(escape_xml("a > b"), "a &gt; b");
    }

    #[test]
    fn test_escape_xml_quotes() {
        assert_eq!(escape_xml("say \"hello\""), "say &quot;hello&quot;");
        assert_eq!(escape_xml("it's"), "it&apos;s");
    }

    #[test]
    fn test_escape_xml_combined() {
        assert_eq!(
            escape_xml("<tag attr=\"val\">content & more</tag>"),
            "&lt;tag attr=&quot;val&quot;&gt;content &amp; more&lt;/tag&gt;"
        );
    }

    #[test]
    fn test_escape_xml_empty() {
        assert_eq!(escape_xml(""), "");
    }

    #[test]
    fn test_escape_xml_no_special_chars() {
        assert_eq!(escape_xml("hello world"), "hello world");
    }

    #[test]
    fn test_write_xml_target_some() {
        let target = Some(EntityLocation {
            file_path: "src/main.rs".to_string(),
            entity_name: "main".to_string(),
            entity_type: "function".to_string(),
            start_line: 1,
            end_line: 10,
            is_public: true,
            content: "fn main() {}".to_string(),
        });

        let mut output = Vec::new();
        write_xml_target(&mut output, &target).unwrap();
        let result = String::from_utf8(output).unwrap();

        assert!(result.contains("<target>"));
        assert!(result.contains("<file>src/main.rs</file>"));
        assert!(result.contains("<name>main</name>"));
        assert!(result.contains("<type>function</type>"));
        assert!(result.contains("start=\"1\""));
        assert!(result.contains("end=\"10\""));
        assert!(result.contains("</target>"));
    }

    #[test]
    fn test_write_xml_target_none() {
        let target: Option<EntityLocation> = None;

        let mut output = Vec::new();
        write_xml_target(&mut output, &target).unwrap();
        let result = String::from_utf8(output).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_write_xml_statistics() {
        let stats = CoveringSetStatistics {
            files_examined: 100,
            files_selected: 10,
            files_excluded: 5,
            entities_selected: 25,
            max_depth_reached: 3,
            limits_reached: false,
        };

        let mut output = Vec::new();
        write_xml_statistics(&mut output, &stats).unwrap();
        let result = String::from_utf8(output).unwrap();

        assert!(result.contains("<statistics>"));
        assert!(result.contains("<files_examined>100</files_examined>"));
        assert!(result.contains("<files_selected>10</files_selected>"));
        assert!(result.contains("<entities_selected>25</entities_selected>"));
        assert!(result.contains("<max_depth_reached>3</max_depth_reached>"));
        assert!(result.contains("<limits_reached>false</limits_reached>"));
        assert!(result.contains("</statistics>"));
    }

    #[test]
    fn test_format_xml_entity() {
        let entity = CoveringSetEntity {
            file_path: "src/lib.rs".to_string(),
            name: "my_func".to_string(),
            entity_type: "function".to_string(),
            start_line: 5,
            end_line: 20,
            distance: 1,
            reason: InclusionReason::TargetFile,
            content: "fn my_func() {}".to_string(),
            references: vec![],
        };

        let result = format_xml_entity(&entity);

        assert!(result.contains("<entity>"));
        assert!(result.contains("<file>src/lib.rs</file>"));
        assert!(result.contains("<name>my_func</name>"));
        assert!(result.contains("<type>function</type>"));
        assert!(result.contains("start=\"5\""));
        assert!(result.contains("end=\"20\""));
        assert!(result.contains("<distance>1</distance>"));
        assert!(result.contains("<content><![CDATA[fn my_func() {}]]></content>"));
        assert!(result.contains("</entity>"));
    }

    #[test]
    fn test_format_xml_file() {
        let file = CoveringSetFile {
            path: "src/main.rs".to_string(),
            distance: 2,
            reason: InclusionReason::DirectDependency,
            importance: None,
            line_ranges: vec![],
        };

        let content = "fn main() { println!(\"Hello\"); }";
        let result = format_xml_file(&file, content);

        assert!(result.contains("<file>"));
        assert!(result.contains("<path>src/main.rs</path>"));
        assert!(result.contains("<distance>2</distance>"));
        assert!(
            result.contains("<content><![CDATA[fn main() { println!(\"Hello\"); }]]></content>")
        );
        assert!(result.contains("</file>"));
    }

    #[test]
    fn test_write_xml_entities() {
        let entities = vec![
            CoveringSetEntity {
                file_path: "a.rs".to_string(),
                name: "func_a".to_string(),
                entity_type: "function".to_string(),
                start_line: 1,
                end_line: 5,
                distance: 0,
                reason: InclusionReason::TargetFile,
                content: "fn func_a() {}".to_string(),
                references: vec![],
            },
            CoveringSetEntity {
                file_path: "b.rs".to_string(),
                name: "func_b".to_string(),
                entity_type: "function".to_string(),
                start_line: 1,
                end_line: 3,
                distance: 1,
                reason: InclusionReason::DirectDependency,
                content: "fn func_b() {}".to_string(),
                references: vec![],
            },
        ];

        let mut output = Vec::new();
        write_xml_entities(&mut output, &entities).unwrap();
        let result = String::from_utf8(output).unwrap();

        assert!(result.contains("<entities count=\"2\">"));
        assert!(result.contains("<name>func_a</name>"));
        assert!(result.contains("<name>func_b</name>"));
        assert!(result.contains("</entities>"));
    }

    #[test]
    fn test_write_xml_files() {
        let files = vec![
            CoveringSetFile {
                path: "a.rs".to_string(),
                distance: 0,
                reason: InclusionReason::TargetFile,
                importance: None,
                line_ranges: vec![],
            },
            CoveringSetFile {
                path: "b.rs".to_string(),
                distance: 1,
                reason: InclusionReason::DirectDependency,
                importance: None,
                line_ranges: vec![],
            },
        ];

        let mut contents = HashMap::new();
        contents.insert("a.rs".to_string(), "// file a".to_string());
        contents.insert("b.rs".to_string(), "// file b".to_string());

        let mut output = Vec::new();
        write_xml_files(&mut output, &files, &contents).unwrap();
        let result = String::from_utf8(output).unwrap();

        assert!(result.contains("<files count=\"2\">"));
        assert!(result.contains("<path>a.rs</path>"));
        assert!(result.contains("<path>b.rs</path>"));
        assert!(result.contains("// file a"));
        assert!(result.contains("// file b"));
        assert!(result.contains("</files>"));
    }

    #[test]
    fn test_write_xml_files_missing_content() {
        let files = vec![CoveringSetFile {
            path: "missing.rs".to_string(),
            distance: 0,
            reason: InclusionReason::TargetFile,
            importance: None,
            line_ranges: vec![],
        }];

        let contents = HashMap::new(); // Empty - no content for the file

        let mut output = Vec::new();
        write_xml_files(&mut output, &files, &contents).unwrap();
        let result = String::from_utf8(output).unwrap();

        assert!(result.contains("<path>missing.rs</path>"));
        assert!(result.contains("<content><![CDATA[]]></content>")); // Empty content
    }
}
