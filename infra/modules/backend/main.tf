# Backend App Service Plan
resource "azurerm_service_plan" "backend" {
  name                = "${var.app_name}-backend-asp"
  resource_group_name = var.resource_group_name
  location            = var.location
  os_type             = "Linux"
  sku_name            = var.app_service_sku_name_backend
  tags                = var.common_tags
  lifecycle {
    ignore_changes = [tags]
  }
}

# Backend App Service
resource "azurerm_linux_web_app" "backend" {
  name                      = "${var.repo_name}-${var.app_env}-api"
  resource_group_name       = var.resource_group_name
  location                  = var.location
  service_plan_id           = azurerm_service_plan.backend.id
  https_only                = true
  virtual_network_subnet_id = var.backend_subnet_id
  identity {
    type = "SystemAssigned"
  }
  site_config {
    always_on                               = true
    container_registry_use_managed_identity = true
    minimum_tls_version                     = "1.3"
    health_check_path                       = "/api/health"
    health_check_eviction_time_in_min       = 2
    application_stack {
      docker_image_name   = var.api_image
      docker_registry_url = var.container_registry_url
    }
    ftps_state = "Disabled"
    cors {
      allowed_origins     = ["*"]
      support_credentials = false
    }
    dynamic "ip_restriction" {
      for_each = split(",", var.frontend_possible_outbound_ip_addresses)
      content {
        ip_address                = ip_restriction.value != "" ? "${ip_restriction.value}/32" : null
        virtual_network_subnet_id = ip_restriction.value == "" ? var.app_service_subnet_id : null
        service_tag               = ip_restriction.value == "" ? "AppService" : null
        action                    = "Allow"
        name                      = "AFInbound${replace(ip_restriction.value, ".", "")}"
        priority                  = 100
      }
    }
    ip_restriction {
      name        = "DenyAll"
      action      = "Deny"
      priority    = 500
      ip_address  = "0.0.0.0/0"
      description = "Deny all other traffic"
    }
    ip_restriction_default_action = "Deny"
  }
  app_settings = {
    NODE_ENV                              = var.node_env
    PORT                                  = "80"
    WEBSITES_PORT                         = "3000"
    DOCKER_ENABLE_CI                      = "true"
    APPLICATIONINSIGHTS_CONNECTION_STRING = var.appinsights_connection_string
    APPINSIGHTS_INSTRUMENTATIONKEY        = var.appinsights_instrumentation_key
    WEBSITE_SKIP_RUNNING_KUDUAGENT        = "false"
    WEBSITES_ENABLE_APP_SERVICE_STORAGE   = "false"
    WEBSITE_ENABLE_SYNC_UPDATE_SITE       = "1"
    IMAGE_TAG                             = var.image_tag
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT             = var.azure_openai_endpoint
    AZURE_OPENAI_API_KEY              = var.azure_openai_api_key
    AZURE_OPENAI_DEPLOYMENT_NAME      = var.azure_openai_deployment_name
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = var.azure_openai_embedding_deployment
    AZURE_OPENAI_LLM_ENDPOINT         = var.azure_openai_llm_endpoint
    AZURE_OPENAI_EMBEDDING_ENDPOINT   = var.azure_openai_embedding_endpoint
    # CosmosDB Configuration
    COSMOS_DB_ENDPOINT               = var.cosmosdb_endpoint
    COSMOS_DB_DATABASE_NAME          = var.cosmosdb_db_name
    COSMOS_DB_CONTAINER_NAME         = var.cosmosdb_container_name
    AZURE_COSMOSDB_DIAGNOSTICS_LEVEL = "debug"
    AZURE_LOG_LEVEL                  = "verbose"
    COSMOS_DB_KEY                    = var.cosmosdb_key

  }
  logs {
    detailed_error_messages = true
    failed_request_tracing  = true
    http_logs {
      file_system {
        retention_in_days = 7
        retention_in_mb   = 100
      }
    }
  }
  tags = var.common_tags
  lifecycle {
    ignore_changes = [tags]
  }
}

# Backend Diagnostics
resource "azurerm_monitor_diagnostic_setting" "backend_diagnostics" {
  name                       = "${var.app_name}-backend-diagnostics"
  target_resource_id         = azurerm_linux_web_app.backend.id
  log_analytics_workspace_id = var.log_analytics_workspace_id
  enabled_log {
    category = "AppServiceHTTPLogs"
  }
  enabled_log {
    category = "AppServiceConsoleLogs"
  }
  enabled_log {
    category = "AppServiceAppLogs"
  }
  enabled_log {
    category = "AppServicePlatformLogs"
  }
}
