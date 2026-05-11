resource "aws_instance" "app" {
  ami = var.ami_id
  instance_type = var.instance_type
  subnet_id = data.aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile = "EC2-green-pulse"
  tags = { Name = "green-pulse-app" }

  root_block_device {
  volume_size = 30   # GB
  volume_type = "gp3"
}

  user_data = <<-EOF
  #!/bin/bash

  #1.Update
  apt-get update -y
  apt-get upgrade -y

  #2.Install dependencies and git
  apt-get install -y ca-certificates curl git

  #3.Setup Docker
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  #4.Add Docker repository
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" > /etc/apt/sources.list.d/docker.list

  #5.Install Docker CE
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

  #6.Enable docker and start
  systemctl enable docker
  systemctl start docker

  #7.Give perm to Ubuntu (no sudo need)
  usermod -aG docker ubuntu

  #8.Clone Repo
  cd /home/ubuntu
  git clone https://github.com/ArinchSup/green-pulse
  chown -R ubuntu:ubuntu /home/ubuntu/green-pulse

  EOF
}
