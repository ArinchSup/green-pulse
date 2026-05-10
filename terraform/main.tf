terraform{
    required_providers {
      aws = {
        source = "hashicorp/aws"
        version = "~> 5.0"
      }
    }

    backend "s3" {
      bucket = "green-pulse-tfstate"
      key = "prod/terraform.tfstate"
      region = "ap-southeast-1"
    }
}

provider "aws" {
  region = "ap-southeast-1"
}