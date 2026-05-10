output "Public_IP" {
  value = aws_instance.app.public_ip
}

output "Instance_ID" {
  value = aws_instance.app.id
}