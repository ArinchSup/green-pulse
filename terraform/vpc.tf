data "aws_vpc" "main" {
    id = var.vpc_id
}

data "aws_subnet" "main" {
    id = var.subnet_id
}