package main

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"log"
	"path"
	"sort"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	function "github.com/ArinchSup/green-pulse/src/controller/function"
)

//go:embed migrations/*.sql
var migrationFiles embed.FS

// Any number works, as long as every migrate run uses the same one.
const migrateLockKey = 20260924

func runMigrate() {
	db := function.ConnectDB()
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	if err := applyAll(ctx, db); err != nil {
		log.Fatalf("migrate failed: %v", err)
	}
	log.Println("migrate finished")
}

func applyAll(ctx context.Context, db *pgxpool.Pool) error {
	names, err := fs.Glob(migrationFiles, "migrations/*.sql")
	if err != nil {
		return err
	}
	sort.Strings(names)

	for _, name := range names {
		if err := applyOne(ctx, db, name); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	return nil
}

func applyOne(ctx context.Context, db *pgxpool.Pool, name string) error {
	sqlText, err := migrationFiles.ReadFile(name)
	if err != nil {
		return err
	}
	version := path.Base(name)

	tx, err := db.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", migrateLockKey); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `CREATE SCHEMA IF NOT EXISTS app;
	CREATE TABLE IF NOT EXISTS app.schema_migrations (
		version    text PRIMARY KEY,
		applied_at timestamptz NOT NULL DEFAULT now()
	)`); err != nil {
		return err
	}

	var applied bool
	err = tx.QueryRow(ctx,
		"SELECT EXISTS (SELECT 1 FROM app.schema_migrations WHERE version = $1)", version,
	).Scan(&applied)
	if err != nil {
		return err
	}
	if applied {
		log.Printf("skip %s (already applied)", version)
		return nil
	}

	if _, err := tx.Exec(ctx, string(sqlText)); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx,
		"INSERT INTO app.schema_migrations (version) VALUES ($1)", version,
	); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return err
	}
	log.Printf("applied %s", version)
	return nil
}
