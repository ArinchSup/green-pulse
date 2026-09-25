# ---- build stage: compile a static linux/amd64 binary ----
FROM golang:1.27 AS build
WORKDIR /src

# Dependencies first: this layer is reused until go.mod or go.sum change.
COPY src/controller/go.mod src/controller/go.sum ./
RUN go mod download

COPY src/controller/ ./
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -o /out/greenpulse .

# ---- run stage: only the binary, no shell, no package manager ----
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/greenpulse /greenpulse
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/greenpulse"]
CMD ["server"]