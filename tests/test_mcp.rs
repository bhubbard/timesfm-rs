use std::sync::Arc;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use timesfm::mcp::{
    create_test_forecaster, evaluate_trend, handle_message, run_mcp_server_io, MockForecaster,
};

#[test]
fn test_initialize_handshake() {
    let forecaster = MockForecaster::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    });

    let resp_str = handle_message(&req.to_string(), &forecaster)
        .expect("Expected response for initialize request");
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(resp["result"]["serverInfo"]["name"], "timesfm-mcp");
    assert_eq!(resp["result"]["serverInfo"]["version"], "0.1.0");
    assert!(resp["result"]["capabilities"]["tools"].is_object());
}

#[test]
fn test_ping() {
    let forecaster = MockForecaster::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": "ping-test-42",
        "method": "ping"
    });

    let resp_str = handle_message(&req.to_string(), &forecaster)
        .expect("Expected response for ping");
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], "ping-test-42");
    assert_eq!(resp["result"], json!({}));
}

#[test]
fn test_notification_no_response() {
    let forecaster = MockForecaster::default();
    let req = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });

    let resp = handle_message(&req.to_string(), &forecaster);
    assert!(resp.is_none(), "Notifications must not produce a response");
}

#[test]
fn test_tools_list() {
    let forecaster = MockForecaster::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    });

    let resp_str = handle_message(&req.to_string(), &forecaster)
        .expect("Expected response for tools/list");
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    let tools = resp["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 3);

    let tool_names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();

    assert!(tool_names.contains(&"forecast_univariate"));
    assert!(tool_names.contains(&"forecast_multivariate"));
    assert!(tool_names.contains(&"evaluate_trend"));

    // Check forecast_univariate schema
    let uni = tools.iter().find(|t| t["name"] == "forecast_univariate").unwrap();
    assert!(uni["inputSchema"]["properties"]["context"].is_object());
    assert!(uni["inputSchema"]["properties"]["horizon"].is_object());
    assert!(uni["inputSchema"]["properties"]["return_quantiles"].is_object());

    // Check forecast_multivariate schema
    let multi = tools.iter().find(|t| t["name"] == "forecast_multivariate").unwrap();
    assert!(multi["inputSchema"]["properties"]["contexts"].is_object());
    assert!(multi["inputSchema"]["properties"]["horizon"].is_object());
    assert!(multi["inputSchema"]["properties"]["past_only_covariates"].is_object());
    assert!(multi["inputSchema"]["properties"]["past_future_covariates"].is_object());

    // Check evaluate_trend schema
    let trend = tools.iter().find(|t| t["name"] == "evaluate_trend").unwrap();
    assert!(trend["inputSchema"]["properties"]["series"].is_object());
}

#[test]
fn test_evaluate_trend_direct() {
    // Upward trend
    let series_up = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let res_up = evaluate_trend(&series_up).unwrap();
    assert_eq!(res_up.count, 5);
    assert!((res_up.slope - 1.0).abs() < 1e-4);
    assert_eq!(res_up.direction, "upward");
    assert!(res_up.has_strong_trend);
    assert!((res_up.r_squared - 1.0).abs() < 1e-4);

    // Downward trend
    let series_down = vec![50.0, 40.0, 30.0, 20.0, 10.0];
    let res_down = evaluate_trend(&series_down).unwrap();
    assert_eq!(res_down.direction, "downward");
    assert!(res_down.slope < 0.0);
    assert!(res_down.has_strong_trend);

    // Flat line
    let series_flat = vec![7.0, 7.0, 7.0, 7.0];
    let res_flat = evaluate_trend(&series_flat).unwrap();
    assert_eq!(res_flat.direction, "flat");
    assert_eq!(res_flat.slope, 0.0);
    assert!(!res_flat.has_strong_trend);

    // Empty series error
    assert!(evaluate_trend(&[]).is_err());
}

#[test]
fn test_tools_call_evaluate_trend() {
    let forecaster = MockForecaster::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "evaluate_trend",
            "arguments": {
                "series": [10.0, 20.0, 30.0, 40.0]
            }
        }
    });

    let resp_str = handle_message(&req.to_string(), &forecaster).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["id"], 10);
    assert_eq!(resp["result"]["isError"], false);
    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let trend_val: Value = serde_json::from_str(content_text).unwrap();

    assert_eq!(trend_val["direction"], "upward");
    assert_eq!(trend_val["count"], 4);
    assert!((trend_val["slope"].as_f64().unwrap() - 10.0).abs() < 1e-3);
}

#[test]
fn test_tools_call_forecast_univariate_mock() {
    let forecaster = MockForecaster::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "forecast_univariate",
            "arguments": {
                "context": [1.0, 2.0, 3.0, 4.0, 5.0],
                "horizon": 4,
                "return_quantiles": true
            }
        }
    });

    let resp_str = handle_message(&req.to_string(), &forecaster).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["id"], 11);
    assert_eq!(resp["result"]["isError"], false);

    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let forecast_res: Value = serde_json::from_str(content_text).unwrap();

    let point_forecast = forecast_res["point_forecast"].as_array().unwrap();
    assert_eq!(point_forecast.len(), 4);
    assert_eq!(forecast_res["horizon"], 4);

    let quantiles = forecast_res["quantiles"].as_array().unwrap();
    assert_eq!(quantiles.len(), 4);
    assert_eq!(quantiles[0].as_array().unwrap().len(), 9);
    assert_eq!(forecast_res["quantile_levels"].as_array().unwrap().len(), 9);
}

#[test]
fn test_tools_call_forecast_multivariate_mock() {
    let forecaster = MockForecaster::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "forecast_multivariate",
            "arguments": {
                "contexts": [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]
                ],
                "horizon": 3,
                "return_quantiles": true
            }
        }
    });

    let resp_str = handle_message(&req.to_string(), &forecaster).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["id"], 12);
    assert_eq!(resp["result"]["isError"], false);

    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let forecast_res: Value = serde_json::from_str(content_text).unwrap();

    let forecasts = forecast_res["forecasts"].as_array().unwrap();
    assert_eq!(forecasts.len(), 2);
    assert_eq!(forecasts[0].as_array().unwrap().len(), 3);
    assert_eq!(forecasts[1].as_array().unwrap().len(), 3);
    assert_eq!(forecast_res["num_variates"], 2);
    assert_eq!(forecast_res["horizon"], 3);
}

#[test]
fn test_synthetic_timesfm_forecaster_tools() -> timesfm::Result<()> {
    let forecaster = create_test_forecaster()?;

    // 1. Univariate tool call with real synthetic TimesFM model
    let req_uni = json!({
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "forecast_univariate",
            "arguments": {
                "context": vec![1.0; 32],
                "horizon": 12,
                "return_quantiles": true
            }
        }
    });

    let resp_str = handle_message(&req_uni.to_string(), &forecaster).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["id"], 20);
    assert_eq!(resp["result"]["isError"], false);

    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let forecast_res: Value = serde_json::from_str(content_text).unwrap();
    let point_forecast = forecast_res["point_forecast"].as_array().unwrap();
    assert_eq!(point_forecast.len(), 12);
    let quantiles = forecast_res["quantiles"].as_array().unwrap();
    assert_eq!(quantiles.len(), 12);
    assert_eq!(quantiles[0].as_array().unwrap().len(), 9);

    // 2. Multivariate tool call with covariates
    let req_multi = json!({
        "jsonrpc": "2.0",
        "id": 21,
        "method": "tools/call",
        "params": {
            "name": "forecast_multivariate",
            "arguments": {
                "contexts": [
                    vec![1.0; 24],
                    vec![2.0; 24]
                ],
                "horizon": 6,
                "past_only_covariates": [
                    vec![0.5; 24]
                ],
                "past_future_covariates": [
                    vec![0.8; 24 + 6]
                ],
                "return_quantiles": true
            }
        }
    });

    let resp_str = handle_message(&req_multi.to_string(), &forecaster).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["id"], 21);
    assert_eq!(resp["result"]["isError"], false);

    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let multi_res: Value = serde_json::from_str(content_text).unwrap();
    let forecasts = multi_res["forecasts"].as_array().unwrap();
    assert_eq!(forecasts.len(), 2);
    assert_eq!(forecasts[0].as_array().unwrap().len(), 6);
    assert_eq!(forecasts[1].as_array().unwrap().len(), 6);

    Ok(())
}

#[test]
fn test_error_handling() {
    let forecaster = MockForecaster::default();

    // 1. Invalid JSON parse error
    let resp_str = handle_message("not valid json", &forecaster).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["error"]["code"], -32700);

    // 2. Unknown method
    let req_unknown = json!({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "unknown_method"
    });
    let resp_str = handle_message(&req_unknown.to_string(), &forecaster).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["error"]["code"], -32601);

    // 3. Unknown tool
    let req_bad_tool = json!({
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {
            "name": "non_existent_tool",
            "arguments": {}
        }
    });
    let resp_str = handle_message(&req_bad_tool.to_string(), &forecaster).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["result"]["isError"], true);

    // 4. Missing arguments / invalid schema
    let req_bad_args = json!({
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "forecast_univariate",
            "arguments": {
                "context": [] // empty context should error
            }
        }
    });
    let resp_str = handle_message(&req_bad_args.to_string(), &forecaster).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["result"]["isError"], true);
}

#[tokio::test]
async fn test_async_duplex_mcp_server() {
    let forecaster = Arc::new(MockForecaster::default());

    // Create duplex streams: (client_read, server_write) and (server_read, client_write)
    let (client_write, server_read) = tokio::io::duplex(4096);
    let (server_write, client_read) = tokio::io::duplex(4096);

    let server_handle = tokio::spawn(async move {
        run_mcp_server_io(forecaster, server_read, server_write)
            .await
            .unwrap();
    });

    let mut client_in = BufReader::new(client_read).lines();
    let mut client_out = client_write;

    // 1. Send initialize
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    });
    client_out
        .write_all(format!("{}\n", init_req).as_bytes())
        .await
        .unwrap();
    client_out.flush().await.unwrap();

    let init_resp_line = client_in.next_line().await.unwrap().unwrap();
    let init_resp: Value = serde_json::from_str(&init_resp_line).unwrap();
    assert_eq!(init_resp["id"], 1);
    assert_eq!(init_resp["result"]["serverInfo"]["name"], "timesfm-mcp");

    // 2. Send evaluate_trend tool call
    let call_req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "evaluate_trend",
            "arguments": {
                "series": [1.0, 2.0, 3.0, 4.0]
            }
        }
    });
    client_out
        .write_all(format!("{}\n", call_req).as_bytes())
        .await
        .unwrap();
    client_out.flush().await.unwrap();

    let call_resp_line = client_in.next_line().await.unwrap().unwrap();
    let call_resp: Value = serde_json::from_str(&call_resp_line).unwrap();
    assert_eq!(call_resp["id"], 2);
    assert_eq!(call_resp["result"]["isError"], false);

    // 3. Close client output to shut down server
    drop(client_out);
    server_handle.await.unwrap();
}
