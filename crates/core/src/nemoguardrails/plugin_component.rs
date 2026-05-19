// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(not(test), allow(dead_code))]

//! NeMo Guardrails plugin component contract.
//!
//! This module defines the Rust-owned configuration contract for the planned
//! built-in `nemoguardrails` plugin. The runtime backend is intentionally not
//! implemented here yet; this surface exists so validation, binding exposure,
//! and later backend work can target one canonical config shape.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use crate::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistrationContext, Result as PluginResult, UnsupportedBehavior, deregister_plugin,
    lookup_plugin, register_plugin,
};

/// The plugin kind name reserved for the planned built-in component.
pub const NEMO_GUARDRAILS_PLUGIN_KIND: &str = "nemoguardrails";

/// Top-level typed helper for configuring the built-in plugin from Rust.
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Whether the component should be activated.
    pub enabled: bool,
    /// NeMo Guardrails component config.
    pub config: NeMoGuardrailsConfig,
}

impl ComponentSpec {
    /// Creates an enabled component spec.
    pub fn new(config: NeMoGuardrailsConfig) -> Self {
        Self {
            enabled: true,
            config,
        }
    }
}

impl From<ComponentSpec> for PluginComponentSpec {
    fn from(value: ComponentSpec) -> Self {
        let Json::Object(config) = serde_json::to_value(value.config)
            .expect("NeMo Guardrails config should serialize to an object")
        else {
            unreachable!("NeMo Guardrails config must serialize to an object");
        };

        PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: value.enabled,
            config,
        }
    }
}

/// Canonical config document for the built-in NeMo Guardrails plugin.
///
/// This surface intentionally keeps the current example plugin's config-source
/// fields while widening the contract for backend mode selection and broader
/// guardrail surfaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeMoGuardrailsConfig {
    /// Config schema version.
    #[serde(default = "default_config_version")]
    pub version: u32,
    /// Backend mode: `remote` or `local`.
    #[serde(default = "default_mode")]
    pub mode: String,
    /// Path to a native NeMo Guardrails config directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    /// Inline native NeMo Guardrails YAML config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_yaml: Option<String>,
    /// Optional inline Colang content. Valid only with `config_yaml`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colang_content: Option<String>,
    /// Provider request/response codec for LLM-managed surfaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
    /// Whether to run input rails around managed LLM execution.
    #[serde(default = "default_true")]
    pub input: bool,
    /// Whether to run output rails around managed LLM execution.
    #[serde(default = "default_true")]
    pub output: bool,
    /// Whether to run tool-input rails around managed tool execution.
    #[serde(default)]
    pub tool_input: bool,
    /// Whether to run tool-output rails around managed tool execution.
    #[serde(default)]
    pub tool_output: bool,
    /// Whether to expose streaming LLM guardrails through the plugin contract.
    #[serde(default)]
    pub streaming: bool,
    /// Intercept priority. Lower values run earlier.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Remote-backend settings used when `mode = "remote"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteBackendConfig>,
    /// Local-backend settings used when `mode = "local"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<LocalBackendConfig>,
    /// Component-local unsupported-config policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for NeMoGuardrailsConfig {
    fn default() -> Self {
        Self {
            version: default_config_version(),
            mode: default_mode(),
            config_path: None,
            config_yaml: None,
            colang_content: None,
            codec: None,
            input: true,
            output: true,
            tool_input: false,
            tool_output: false,
            streaming: false,
            priority: default_priority(),
            remote: None,
            local: None,
            policy: ConfigPolicy::default(),
        }
    }
}

/// Remote-backend settings for a hosted Guardrails service.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RemoteBackendConfig {
    /// Base URL for the remote Guardrails service.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Remote Guardrails config identifier to use for this component.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_id: Option<String>,
    /// Remote Guardrails config identifiers to combine for this component.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_ids: Vec<String>,
    /// Static request headers sent to the remote service.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout_millis")]
    pub timeout_millis: u64,
}

/// Local-backend settings for the Python `nemoguardrails` runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LocalBackendConfig {
    /// Optional import path for the Python runtime module.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python_module: Option<String>,
}

struct NeMoGuardrailsPlugin;

impl Plugin for NeMoGuardrailsPlugin {
    fn plugin_kind(&self) -> &str {
        NEMO_GUARDRAILS_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        validate_nemoguardrails_plugin_config(plugin_config)
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        Box::pin(async {
            Err(PluginError::RegistrationFailed(
                "built-in NeMo Guardrails plugin backend is not implemented yet".to_string(),
            ))
        })
    }
}

/// Registers the `nemoguardrails` component kind in the plugin registry.
pub fn register_nemoguardrails_component() -> PluginResult<()> {
    match register_plugin(Arc::new(NeMoGuardrailsPlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(message))
            if message.contains("already registered")
                && lookup_plugin(NEMO_GUARDRAILS_PLUGIN_KIND).is_some() =>
        {
            Ok(())
        }
        Err(err) => Err(err),
    }
}

/// Deregisters the `nemoguardrails` component kind from the plugin registry.
pub fn deregister_nemoguardrails_component() -> bool {
    deregister_plugin(NEMO_GUARDRAILS_PLUGIN_KIND)
}

fn parse_nemoguardrails_config(
    plugin_config: &Map<String, Json>,
) -> PluginResult<NeMoGuardrailsConfig> {
    serde_json::from_value(Json::Object(plugin_config.clone())).map_err(|err| {
        PluginError::InvalidConfig(format!("invalid NeMo Guardrails plugin config: {err}"))
    })
}

fn validate_nemoguardrails_plugin_config(
    plugin_config: &Map<String, Json>,
) -> Vec<ConfigDiagnostic> {
    let config = match parse_nemoguardrails_config(plugin_config) {
        Ok(config) => config,
        Err(err) => {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "nemoguardrails.invalid_plugin_config".to_string(),
                component: Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                field: None,
                message: err.to_string(),
            }];
        }
    };

    let mut diagnostics = vec![];

    validate_unknown_fields(
        &mut diagnostics,
        &config.policy,
        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
        plugin_config,
        &[
            "version",
            "mode",
            "config_path",
            "config_yaml",
            "colang_content",
            "codec",
            "input",
            "output",
            "tool_input",
            "tool_output",
            "streaming",
            "priority",
            "remote",
            "local",
            "policy",
        ],
    );

    validate_policy_fields(&mut diagnostics, &config.policy, plugin_config);
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "remote",
        &[
            "endpoint",
            "config_id",
            "config_ids",
            "headers",
            "timeout_millis",
        ],
    );
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "local",
        &["python_module"],
    );

    validate_version(&mut diagnostics, &config.policy, config.version);
    validate_mode(&mut diagnostics, &config.policy, &config.mode);
    validate_config_source(&mut diagnostics, &config.policy, &config);
    validate_non_empty_strings(&mut diagnostics, &config.policy, &config);
    validate_codec_requirements(&mut diagnostics, &config.policy, &config);
    validate_surface_selection(&mut diagnostics, &config.policy, &config);
    validate_backend_sections(&mut diagnostics, &config.policy, &config);

    diagnostics
}

fn validate_version(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, version: u32) {
    if version != 1 {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_config_version",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("version".to_string()),
            format!("NeMo Guardrails config version {version} is unsupported"),
        );
    }
}

fn validate_mode(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, mode: &str) {
    if !matches!(mode, "remote" | "local") {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("mode".to_string()),
            "mode must be 'remote' or 'local'".to_string(),
        );
    }
}

fn validate_config_source(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    let has_config_path = config
        .config_path
        .as_ref()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let has_config_yaml = config
        .config_yaml
        .as_ref()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);

    if has_config_path == has_config_yaml && config.mode == "local" {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.invalid_config_source",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            None,
            "exactly one of config_path or config_yaml is required in local mode".to_string(),
        );
    }

    if config.colang_content.is_some() && !has_config_yaml {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("colang_content".to_string()),
            "colang_content can only be used with config_yaml".to_string(),
        );
    }

    if config.mode == "remote" {
        let has_remote_config_id = config
            .remote
            .as_ref()
            .and_then(|remote| remote.config_id.as_ref())
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false);
        let has_remote_config_ids = config
            .remote
            .as_ref()
            .map(|remote| {
                remote
                    .config_ids
                    .iter()
                    .any(|value| !value.trim().is_empty())
            })
            .unwrap_or(false);

        if has_remote_config_id && has_remote_config_ids {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemoguardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("remote".to_string()),
                "remote.config_id and remote.config_ids cannot be used together".to_string(),
            );
        }

        if !(has_config_path || has_config_yaml || has_remote_config_id || has_remote_config_ids) {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemoguardrails.invalid_config_source",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                None,
                "remote mode requires config_path, config_yaml, remote.config_id, or remote.config_ids"
                    .to_string(),
            );
        }
    }
}

fn validate_codec_requirements(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    let llm_surface_enabled = config.input || config.output || config.streaming;
    if !llm_surface_enabled {
        return;
    }

    let Some(codec) = config.codec.as_deref() else {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("codec".to_string()),
            "codec is required when any LLM surface is enabled".to_string(),
        );
        return;
    };

    if !matches!(
        codec,
        "openai_chat" | "openai_responses" | "anthropic_messages"
    ) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("codec".to_string()),
            "codec must be 'openai_chat', 'openai_responses', or 'anthropic_messages'".to_string(),
        );
    }
}

fn validate_non_empty_strings(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    if let Some(config_path) = &config.config_path
        && config_path.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("config_path".to_string()),
            "config_path must not be empty".to_string(),
        );
    }

    if let Some(config_yaml) = &config.config_yaml
        && config_yaml.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("config_yaml".to_string()),
            "config_yaml must not be empty".to_string(),
        );
    }

    if let Some(colang_content) = &config.colang_content
        && colang_content.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("colang_content".to_string()),
            "colang_content must not be empty".to_string(),
        );
    }

    if config.mode == "remote" {
        match &config.remote {
            Some(remote)
                if remote
                    .endpoint
                    .as_ref()
                    .is_some_and(|value| !value.trim().is_empty()) => {}
            _ => push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemoguardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("remote.endpoint".to_string()),
                "remote.endpoint is required when mode is 'remote'".to_string(),
            ),
        }

        if let Some(config_id) = config
            .remote
            .as_ref()
            .and_then(|remote| remote.config_id.as_ref())
            && config_id.trim().is_empty()
        {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemoguardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("remote.config_id".to_string()),
                "remote.config_id must not be empty".to_string(),
            );
        }

        if let Some(remote) = &config.remote {
            for (index, config_id) in remote.config_ids.iter().enumerate() {
                if config_id.trim().is_empty() {
                    push_policy_diag(
                        diagnostics,
                        policy.unsupported_value,
                        "nemoguardrails.unsupported_value",
                        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                        Some(format!("remote.config_ids[{index}]")),
                        "remote.config_ids entries must not be empty".to_string(),
                    );
                }
            }
        }
    }
}

fn validate_surface_selection(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    if config.input || config.output || config.tool_input || config.tool_output || config.streaming
    {
        return;
    }

    push_policy_diag(
        diagnostics,
        policy.unsupported_value,
        "nemoguardrails.unsupported_value",
        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
        None,
        "at least one Guardrails surface must be enabled".to_string(),
    );
}

fn validate_backend_sections(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    match config.mode.as_str() {
        "remote" if config.local.is_some() => push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("local".to_string()),
            "local backend settings cannot be used when mode is 'remote'".to_string(),
        ),
        "local" if config.remote.is_some() => push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemoguardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("remote".to_string()),
            "remote backend settings cannot be used when mode is 'local'".to_string(),
        ),
        _ => {}
    }
}

fn validate_policy_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
) {
    if let Some(policy_json) = plugin_config.get("policy").and_then(Json::as_object) {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some("policy".to_string()),
            policy_json,
            &["unknown_component", "unknown_field", "unsupported_value"],
        );
    }
}

fn validate_section_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    section: &str,
    known_fields: &[&str],
) {
    if let Some(section_json) = plugin_config.get(section).and_then(Json::as_object) {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some(section.to_string()),
            section_json,
            known_fields,
        );
    }
}

fn validate_unknown_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    component: Option<String>,
    config: &Map<String, Json>,
    known_fields: &[&str],
) {
    for field in config.keys() {
        if !known_fields.contains(&field.as_str()) {
            push_policy_diag(
                diagnostics,
                policy.unknown_field,
                "nemoguardrails.unknown_field",
                component.clone(),
                Some(field.clone()),
                format!(
                    "field '{}' is not recognized for '{}'",
                    field,
                    component.as_deref().unwrap_or("unknown")
                ),
            );
        }
    }
}

fn push_policy_diag(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
    component: Option<String>,
    field: Option<String>,
    message: String,
) {
    let level = match behavior {
        UnsupportedBehavior::Ignore => return,
        UnsupportedBehavior::Warn => DiagnosticLevel::Warning,
        UnsupportedBehavior::Error => DiagnosticLevel::Error,
    };
    diagnostics.push(ConfigDiagnostic {
        level,
        code: code.to_string(),
        component,
        field,
        message,
    });
}

fn default_config_version() -> u32 {
    1
}

fn default_mode() -> String {
    "remote".to_string()
}

fn default_true() -> bool {
    true
}

fn default_priority() -> i32 {
    100
}

fn default_timeout_millis() -> u64 {
    30_000
}

#[cfg(test)]
#[path = "../../tests/unit/nemoguardrails/plugin_component_tests.rs"]
mod tests;
