variable "aws_region" {
  description = "The region"
  default = "ap-southeast-1"
}

variable "instance_type" {
  description = "Instance type: m7i-flex.large"
  default = "m7i-flex.large"
}

variable "ami_id" {
  description = "Ami id"
  default = "ami-0e7ff22101b84bcff"
}