#BUILD
FROM golang:1.23-alphine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
#Compile to binary
RUN CGO_ENABLED=0 GOOS=linux go build -o server .

FROM alphine:latest

WORKDIR /app

COPY --from=builder /app/server .

EXPOSE 8080

CMD ["./server"]