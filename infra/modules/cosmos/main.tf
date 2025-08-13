resource "azurerm_cosmosdb_account" "cosmosdb_sql" {
  name                          = "${var.app_name}-cosmosdb-sql"
  location                      = var.location
  resource_group_name           = var.resource_group_name
  offer_type                    = "Standard"
  kind                          = "GlobalDocumentDB"
  public_network_access_enabled = false

  # Enable vector search capability
  capabilities {
    name = "EnableNoSQLVectorSearch"
  }

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = "canadacentral"
    failover_priority = 0
  }
  tags = var.common_tags
  lifecycle {
    ignore_changes = [tags]
  }
}

resource "azurerm_monitor_diagnostic_setting" "cosmosdb_sql_diagnostics" {
  name                       = "${var.app_name}-cosmosdb-diagnostics"
  target_resource_id         = azurerm_cosmosdb_account.cosmosdb_sql.id
  log_analytics_workspace_id = var.log_analytics_workspace_id

  enabled_log {
    category = "DataPlaneRequests"
  }
  enabled_log {
    category = "MongoRequests"
  }
  enabled_log {
    category = "QueryRuntimeStatistics"
  }
  enabled_log {
    category = "PartitionKeyRUConsumption"
  }
  enabled_log {
    category = "ControlPlaneRequests"
  }

}


resource "azurerm_cosmosdb_sql_database" "cosmosdb_sql_db" {
  name                = var.cosmosdb_sql_database_name
  account_name        = azurerm_cosmosdb_account.cosmosdb_sql.name
  resource_group_name = var.resource_group_name
  autoscale_settings {
    max_throughput = 4000
  }
}

resource "azurerm_cosmosdb_sql_container" "cosmosdb_sql_db_container" {
  name                = var.cosmosdb_sql_database_container_name
  resource_group_name = var.resource_group_name
  account_name        = azurerm_cosmosdb_account.cosmosdb_sql.name
  database_name       = azurerm_cosmosdb_sql_database.cosmosdb_sql_db.name
  partition_key_paths = ["/partitionKey"]

  # Optimized indexing policy for document storage with embeddings
  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }

    excluded_path {
      path = "/embedding/?"
    }

    # Composite index for partitionKey + type queries (optimizes getAllDocuments performance)
    composite_index {
      index {
        path  = "/partitionKey"
        order = "ascending"
      }
      index {
        path  = "/type"
        order = "ascending"
      }
    }

    # Composite index for documentId + partitionKey + type queries (optimizes chunk retrieval)
    composite_index {
      index {
        path  = "/documentId"
        order = "ascending"
      }
      index {
        path  = "/partitionKey"
        order = "ascending"
      }
      index {
        path  = "/type"
        order = "ascending"
      }
    }

    # Index for uploadedAt for sorting (optional but recommended for admin queries)
    composite_index {
      index {
        path  = "/uploadedAt"
        order = "descending"
      }
      index {
        path  = "/type"
        order = "ascending"
      }
    }
  }
}

# Vector indexing policy using AzAPI provider for advanced vector search capabilities
resource "azapi_update_resource" "cosmosdb_container_vector_policy" {
  type        = "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15"
  resource_id = azurerm_cosmosdb_sql_container.cosmosdb_sql_db_container.id

  body = {
    properties = {
      resource = {
        vectorEmbeddingPolicy = {
          vectorEmbeddings = [
            {
              path             = "/embedding"
              dataType         = "float32"
              dimensions       = 3072
              distanceFunction = "cosine"
            }
          ]
        }
        indexingPolicy = {
          indexingMode = "consistent"
          automatic    = true
          includedPaths = [
            {
              path = "/*"
            }
          ]
          excludedPaths = [
            {
              path = "/embedding/?"
            }
          ]
          compositeIndexes = [
            [
              {
                path  = "/partitionKey"
                order = "ascending"
              },
              {
                path  = "/type"
                order = "ascending"
              }
            ],
            [
              {
                path  = "/documentId"
                order = "ascending"
              },
              {
                path  = "/partitionKey"
                order = "ascending"
              },
              {
                path  = "/type"
                order = "ascending"
              }
            ],
            [
              {
                path  = "/uploadedAt"
                order = "descending"
              },
              {
                path  = "/type"
                order = "ascending"
              }
            ]
          ]
          vectorIndexes = [
            {
              path = "/embedding"
              type = "diskANN"
            }
          ]
        }
      }
    }
  }

  depends_on = [azurerm_cosmosdb_sql_container.cosmosdb_sql_db_container]
}

