// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the built-in NeMo Guardrails plugin component.

use super::*;
use crate::api::runtime::NemoFlowContextState;
use crate::api::runtime::global_context;
use crate::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins,
    list_plugin_kinds, lookup_plugin, validate_plugin_config,
};
use serde_json::Value as Json;
use serde_json::json;

fn reset_runtime() {
    let _ = clear_plugin_configuration();
    let _ = deregister_nemoguardrails_component();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoFlowContextState::new();
}

fn ensure_registered() {
    register_nemoguardrails_component().unwrap();
}

fn component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
        enabled: true,
        config,
    }
}

fn disabled_component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
        enabled: false,
        config,
    }
}

fn plugin_config(config: Json) -> PluginConfig {
    PluginConfig {
        version: 1,
        components: vec![component(config)],
        policy: Default::default(),
    }
}

fn minimal_valid_config() -> Json {
    json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "safety-default",
        },
    })
}

#[test]
fn explicit_registration_makes_kind_available() {
    let _guard = crate::nemoguardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    assert!(!list_plugin_kinds().contains(&NEMO_GUARDRAILS_PLUGIN_KIND.to_string()));
    ensure_registered();
    assert!(list_plugin_kinds().contains(&NEMO_GUARDRAILS_PLUGIN_KIND.to_string()));
    assert!(lookup_plugin(NEMO_GUARDRAILS_PLUGIN_KIND).is_some());
    assert!(!validate_plugin_config(&plugin_config(minimal_valid_config())).has_errors());
}

#[test]
fn disabled_component_validates_and_initializes_without_runtime_work() {
    let _guard = crate::nemoguardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let config = PluginConfig {
        version: 1,
        components: vec![disabled_component(minimal_valid_config())],
        policy: Default::default(),
    };
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins(config)).unwrap();
}

#[test]
fn duplicate_component_is_rejected_as_singleton() {
    let _guard = crate::nemoguardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let config = PluginConfig {
        version: 1,
        components: vec![
            component(minimal_valid_config()),
            component(minimal_valid_config()),
        ],
        policy: Default::default(),
    };
    let report = validate_plugin_config(&config);
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "plugin.duplicate_component")
    );
}

#[test]
fn invalid_shapes_and_values_are_reported() {
    let _guard = crate::nemoguardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let invalid_shape = validate_plugin_config(&plugin_config(json!({
        "version": "one",
    })));
    assert!(invalid_shape.has_errors());
    assert!(
        invalid_shape
            .diagnostics
            .iter()
            .any(|diag| diag.code == "nemoguardrails.invalid_plugin_config")
    );

    let invalid_sources = validate_plugin_config(&plugin_config(json!({
        "mode": "edge",
        "config_path": "./rails",
        "config_yaml": "models: []",
        "input": false,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "streaming": false
    })));
    assert!(invalid_sources.has_errors());
    assert!(
        invalid_sources
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("mode"))
    );
    assert!(
        invalid_sources
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("mode"))
    );

    let report = validate_plugin_config(&plugin_config(json!({
        "config_path": "./rails",
        "colang_content": "define flow x",
        "input": false,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "streaming": false
    })));
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.message.contains("colang_content can only be used"))
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("remote.endpoint"))
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.message.contains("at least one Guardrails surface"))
    );

    let remote_missing_config = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000"},
    })));
    assert!(remote_missing_config.has_errors());
    assert!(
        remote_missing_config
            .diagnostics
            .iter()
            .any(|diag| diag.message.contains("remote mode requires"))
    );

    let remote_conflicting_ids = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "one",
            "config_ids": ["two"]
        },
    })));
    assert!(remote_conflicting_ids.has_errors());
    assert!(remote_conflicting_ids.diagnostics.iter().any(|diag| {
        diag.message
            .contains("config_id and remote.config_ids cannot be used together")
    }));

    let local_missing_source = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "codec": "openai_chat",
    })));
    assert!(local_missing_source.has_errors());
    assert!(local_missing_source.diagnostics.iter().any(|diag| {
        diag.message
            .contains("exactly one of config_path or config_yaml is required in local mode")
    }));
}

#[test]
fn unknown_fields_and_backend_conflicts_follow_policy() {
    let _guard = crate::nemoguardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let warn_report = validate_plugin_config(&plugin_config(json!({
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000", "config_id": "safety-default"},
        "bogus": true,
        "local": {"python_module": "nemoguardrails"},
    })));
    assert!(warn_report.has_errors());
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "nemoguardrails.unknown_field")
    );
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("local"))
    );

    let ignored = validate_plugin_config(&plugin_config(json!({
        "policy": {"unknown_field": "ignore", "unsupported_value": "ignore"},
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000", "config_id": "safety-default"},
        "bogus": true,
        "local": {"python_module": "nemoguardrails"},
    })));
    assert!(!ignored.has_errors());
    assert!(ignored.diagnostics.is_empty());
}

#[test]
fn enabled_initialization_fails_fast_until_backend_exists() {
    let _guard = crate::nemoguardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    ensure_registered();

    let error =
        futures::executor::block_on(initialize_plugins(plugin_config(minimal_valid_config())))
            .unwrap_err();

    match error {
        crate::plugin::PluginError::RegistrationFailed(message) => {
            assert!(message.contains("not implemented yet"));
        }
        other => panic!("unexpected error: {other}"),
    }
}
