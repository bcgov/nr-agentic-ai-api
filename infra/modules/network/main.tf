data "azurerm_virtual_network" "main" {
  name                = var.vnet_name
  resource_group_name = var.vnet_resource_group_name
}

# NSG for privateendpoints subnet
resource "azurerm_network_security_group" "privateendpoints" {
  name                = "${var.resource_group_name}-pe-nsg"
  location            = var.location
  resource_group_name = var.vnet_resource_group_name

  security_rule {
    name                       = "AllowInboundFromApp"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.app_service_subnet_cidr
    destination_address_prefix = local.private_endpoints_subnet_cidr
    destination_port_range     = "*"
    source_port_range          = "*"
  }

  security_rule {
    name                       = "AllowOutboundToApp"
    priority                   = 101
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "*"
    destination_address_prefix = local.app_service_subnet_cidr
    source_address_prefix      = local.private_endpoints_subnet_cidr
    source_port_range          = "*"
    destination_port_range     = "*"
  }
  security_rule {
    name                       = "AllowInboundFromContainerInstance"
    priority                   = 104
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.container_instance_subnet_cidr
    destination_address_prefix = local.private_endpoints_subnet_cidr
    destination_port_range     = "*"
    source_port_range          = "*"
  }

  security_rule {
    name                       = "AllowOutboundToContainerInstance"
    priority                   = 105
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "*"
    destination_address_prefix = local.container_instance_subnet_cidr
    source_address_prefix      = local.private_endpoints_subnet_cidr
    source_port_range          = "*"
    destination_port_range     = "*"
  }
  security_rule {
    name                       = "AllowInboundFromWeb"
    priority                   = 102
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.web_subnet_cidr
    source_port_range          = "*"
    destination_address_prefix = local.private_endpoints_subnet_cidr
    destination_port_range     = "*"
  }

  security_rule {
    name                       = "AllowOutboundToWeb"
    priority                   = 103
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "*"
    destination_address_prefix = local.web_subnet_cidr
    source_address_prefix      = local.private_endpoints_subnet_cidr
    source_port_range          = "*"
    destination_port_range     = "*"
  }
  tags = var.common_tags
  lifecycle {
    ignore_changes = [
      tags
    ]
  }
}

# NSG for app service subnet
resource "azurerm_network_security_group" "app_service" {
  name                = "${var.resource_group_name}-as-nsg"
  location            = var.location
  resource_group_name = var.vnet_resource_group_name

  security_rule {
    name                       = "AllowAppFromPrivateEndpoint"
    priority                   = 102
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.private_endpoints_subnet_cidr
    source_port_range          = "*"
    destination_address_prefix = local.app_service_subnet_cidr
    destination_port_range     = "*"
  }

  security_rule {
    name                       = "AllowAppToPrivateEndpoint"
    priority                   = 103
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "*"
    destination_address_prefix = local.private_endpoints_subnet_cidr
    source_address_prefix      = local.app_service_subnet_cidr
    source_port_range          = "*"
    destination_port_range     = "*"
  }
  security_rule {
    name                       = "AllowAppFromContainerInstance"
    priority                   = 104
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.container_instance_subnet_cidr
    source_port_range          = "*"
    destination_address_prefix = local.app_service_subnet_cidr
    destination_port_range     = "*"
  }

  security_rule {
    name                       = "AllowAppToContainerInstance"
    priority                   = 105
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "*"
    destination_address_prefix = local.container_instance_subnet_cidr
    destination_port_range     = "*"
    source_address_prefix      = local.app_service_subnet_cidr
    source_port_range          = "*"
  }
  security_rule {
    name                       = "AllowAppFromWeb"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_address_prefix      = local.web_subnet_cidr
    destination_port_range     = "*"
    source_port_range          = "*"
    destination_address_prefix = local.app_service_subnet_cidr
  }

  security_rule {
    name                       = "AllowAppToWeb"
    priority                   = 101
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    destination_address_prefix = local.web_subnet_cidr
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = local.app_service_subnet_cidr
  }

  security_rule {
    name                       = "AllowAppFromInternet"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_address_prefix      = "*"
    source_port_range          = "*"
    destination_address_prefix = local.app_service_subnet_cidr
    destination_port_ranges    = ["80", "443"]
  }
  security_rule {
    name                       = "AllowAppOutboundToInternet"
    priority                   = 120
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_address_prefix      = local.app_service_subnet_cidr
    destination_address_prefix = "*"
    source_port_range          = "*"
    destination_port_ranges    = ["80", "443"]
  }
  tags = var.common_tags
  lifecycle {
    ignore_changes = [
      tags
    ]
  }
}

# NSG for web subnet
resource "azurerm_network_security_group" "web" {
  name                = "${var.resource_group_name}-web-nsg"
  location            = var.location
  resource_group_name = var.vnet_resource_group_name

  security_rule {
    name                       = "AllowHTTPFromInternet"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_address_prefix      = "*"
    source_port_range          = "*"
    destination_address_prefix = local.web_subnet_cidr
    destination_port_ranges    = ["80", "443"]
  }
  security_rule {
    name                       = "AllowOutboundToInternet"
    priority                   = 120
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_address_prefix      = local.web_subnet_cidr
    destination_address_prefix = "*"
    source_port_range          = "*"
    destination_port_ranges    = ["80", "443"]
  }
  security_rule {
    name                       = "AllowOutboundToApp"
    priority                   = 101
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "*"
    destination_address_prefix = local.app_service_subnet_cidr
    source_address_prefix      = local.web_subnet_cidr
    source_port_range          = "*"
    destination_port_ranges    = ["3000-9000"]
  }
  security_rule {
    name                       = "AllowInboundFromAppService"
    priority                   = 104
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.app_service_subnet_cidr
    destination_address_prefix = local.web_subnet_cidr
    source_port_range          = "*"
    destination_port_ranges    = ["3000-9000"]
  }

  security_rule {
    name                       = "AllowOutboundToContainerInstance"
    priority                   = 102
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "*"
    destination_address_prefix = local.container_instance_subnet_cidr
    source_address_prefix      = local.web_subnet_cidr
    source_port_range          = "*"
    destination_port_ranges    = ["3000-9000"]
  }
  security_rule {
    name                       = "AllowInboundFromContainerInstance"
    priority                   = 105
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.container_instance_subnet_cidr
    destination_address_prefix = local.web_subnet_cidr
    source_port_range          = "*"
    destination_port_ranges    = ["3000-9000"]
  }
  tags = var.common_tags
  lifecycle {
    ignore_changes = [
      tags
    ]
  }
}
resource "azurerm_network_security_group" "container_instance" {
  name                = "${var.resource_group_name}-ci-nsg"
  location            = var.location
  resource_group_name = var.vnet_resource_group_name

  security_rule {
    name                       = "AllowInboundFromAppService"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.app_service_subnet_cidr
    destination_address_prefix = local.container_instance_subnet_cidr
    source_port_ranges         = ["3000-9000"]
    destination_port_ranges    = ["3000-9000"]
  }

  security_rule {
    name                       = "AllowOutboundToAppService"
    priority                   = 101
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "*"
    destination_address_prefix = local.app_service_subnet_cidr
    source_address_prefix      = local.container_instance_subnet_cidr
    source_port_ranges         = ["3000-9000"]
    destination_port_ranges    = ["3000-9000"]
  }

  security_rule {
    name                       = "AllowInboundFromWeb"
    priority                   = 102
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    source_address_prefix      = local.web_subnet_cidr
    destination_address_prefix = local.web_subnet_cidr
    destination_port_ranges    = ["3000-9000"]
  }

  security_rule {
    name                       = "AllowOutboundToWeb"
    priority                   = 103
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    destination_address_prefix = local.web_subnet_cidr
    source_address_prefix      = local.container_instance_subnet_cidr
    source_port_range          = "*"
    destination_port_ranges    = ["3000-9000"]
  }
  # Allow inbound from Private Endpoints subnet to Container Instances subnet
  security_rule {
    name                       = "AllowInboundFromPrivateEndpoint"
    priority                   = 106
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.private_endpoints_subnet_cidr
    destination_address_prefix = local.container_instance_subnet_cidr
    source_port_range          = "*"
    destination_port_range     = "*"
  }

  # Allow outbound to Private Endpoints subnet from Container Instances subnet
  security_rule {
    name                       = "AllowOutboundToPrivateEndpoint"
    priority                   = 107
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "*"
    source_address_prefix      = local.container_instance_subnet_cidr
    destination_address_prefix = local.private_endpoints_subnet_cidr
    source_port_range          = "*"
    destination_port_range     = "*"
  }
  security_rule {
    name                       = "AllowInboundFromInternet"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_address_prefix      = "*"
    destination_address_prefix = local.container_instance_subnet_cidr
    source_port_range          = "*"
    destination_port_ranges    = ["80", "443"]
  }

  security_rule {
    name                       = "AllowOutboundToInternet"
    priority                   = 120
    direction                  = "Outbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    destination_address_prefix = "*"
    source_address_prefix      = local.container_instance_subnet_cidr
    source_port_range          = "*"
    destination_port_ranges    = ["80", "443"]
  }
  tags = var.common_tags
  lifecycle {
    ignore_changes = [
      tags
    ]
  }
}

# Subnets
resource "azapi_resource" "privateendpoints_subnet" {
  type      = "Microsoft.Network/virtualNetworks/subnets@2023-04-01"
  name      = var.private_endpoint_subnet_name
  parent_id = data.azurerm_virtual_network.main.id
  locks     = [data.azurerm_virtual_network.main.id]
  body = {
    properties = {
      addressPrefix = local.private_endpoints_subnet_cidr
      networkSecurityGroup = {
        id = azurerm_network_security_group.privateendpoints.id
      }
    }
  }
  response_export_values = ["*"]
}

resource "azapi_resource" "app_service_subnet" {
  type      = "Microsoft.Network/virtualNetworks/subnets@2023-04-01"
  name      = var.apps_service_subnet_name
  parent_id = data.azurerm_virtual_network.main.id
  locks     = [data.azurerm_virtual_network.main.id]
  body = {
    properties = {
      addressPrefix = local.app_service_subnet_cidr
      networkSecurityGroup = {
        id = azurerm_network_security_group.app_service.id
      }
      delegations = [
        {
          name = "app-service-delegation"
          properties = {
            serviceName = "Microsoft.Web/serverFarms"
          }
        }
      ]
    }
  }
  response_export_values = ["*"]
}

resource "azapi_resource" "container_instance_subnet" {
  type      = "Microsoft.Network/virtualNetworks/subnets@2023-04-01"
  name      = var.container_instance_subnet_name
  parent_id = data.azurerm_virtual_network.main.id
  locks     = [data.azurerm_virtual_network.main.id]
  body = {
    properties = {
      addressPrefix = local.container_instance_subnet_cidr
      networkSecurityGroup = {
        id = azurerm_network_security_group.container_instance.id
      }
      delegations = [
        {
          name = "aci-delegation"
          properties = {
            serviceName = "Microsoft.ContainerInstance/containerGroups"
          }
        }
      ]
    }
  }
  response_export_values = ["*"]
}

resource "azapi_resource" "web_subnet" {
  type      = "Microsoft.Network/virtualNetworks/subnets@2023-04-01"
  name      = var.web_subnet_name
  parent_id = data.azurerm_virtual_network.main.id
  locks     = [data.azurerm_virtual_network.main.id]
  body = {
    properties = {
      addressPrefix = local.web_subnet_cidr
      networkSecurityGroup = {
        id = azurerm_network_security_group.web.id
      }
    }
  }
  response_export_values = ["*"]
}
