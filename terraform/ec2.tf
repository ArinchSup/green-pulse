resource "aws_instance" "app" {
  ami = var.ami_id
  instance_type = var.instance_type
  subnet_id = data.aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile = "EC2-green-pulse"
  tags = { Name = "green-pulse-app" }
}