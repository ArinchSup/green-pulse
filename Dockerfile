# ---- build stage: compile a static linux/amd64 binary ----
FROM golang:1.27 AS build
WORKDIR /src

# Dependencies first: this layer is reused until go.mod or go.sum change.
COPY backend/go.* ./
RUN go mod download

COPY backend/ ./
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -o /out/green-pulse ./cmd/green-pulse


# ---- run stage: only the binary, no shell, no package manager ----
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/green-pulse /green-pulse
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/green-pulse"]
CMD ["server"]