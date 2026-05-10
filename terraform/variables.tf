variable "aws_region" {
  description = "The region"
  default = "ap-southeast-1"
}

variable "instance_type" {
  description = "Instance type: m7i-flex.large"
  default = "m7i-flex.large"
}

variable "ami_id" {
  description = "ami id"
  default = "ami-0e7ff22101b84bcff"
}

variable "vpc_id" {
  description = "VPC-id"
  default = "vpc-076836b55613abe84"
}

variable "subnet_id" {
  description = "Subnet"
  default = "subnet-0cb56148225618e71"
}

variable "my_ip" {
  description = "My private IP"
  default = "110.164.200.203/32"
}