variable "app_name" {
  description = "Name of the application"
  type        = string
  nullable    = false
}


variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  nullable    = false
}


variable "location" {
  description = "Azure region for resources"
  type        = string
  nullable    = false
}

variable "private_endpoint_subnet_id" {
  description = "The ID of the subnet for the private endpoint."
  type        = string
  nullable    = false
}

variable "resource_group_name" {
  description = "The name of the resource group to create."
  type        = string
  nullable    = false
}

variable "cosmosdb_sql_database_container_name" {
  type        = string
  default     = "cosmosContainer"
  description = "Name of the Cosmos DB SQL database container."
}


variable "cosmosdb_sql_database_name" {
  type        = string
  default     = "cosmosDatabase"
  description = "Name of the Cosmos DB SQL database."
}

variable "log_analytics_workspace_id" {
  description = "The resource ID of the Log Analytics workspace for diagnostics."
  type        = string
  nullable    = false
}

variable "embedding_dimensions" {
  description = "Dimensions for the vector embeddings (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)"
  type        = number
  default     = 3072
  validation {
    condition     = contains([1536, 3072], var.embedding_dimensions)
    error_message = "Embedding dimensions must be either 1536 (text-embedding-3-small) or 3072 (text-embedding-3-large)."
  }
}
