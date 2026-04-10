package main

import (
	"fmt"
	"log"
	"os/exec"
	"strings"

	"github.com/ArinchSup/green-pulse/src/database/db"
)

func fetchAndInsert(symbol string) error {
	cmd := exec.Command("python", "./scripts/fetch.py", symbol)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return err
	}
	log.Println(string(out))

	path := "./data/raw/" + symbol + ".csv"
	return db.InsertFromCSV(symbol, path)
}

func printline() {
	fmt.Println("====================================")
}

func printMenu() {
	printline()
	fmt.Println("Type option number")
	fmt.Println("0. exit")
	fmt.Println("1. access to information")
	fmt.Println("2. add stock to db")
	fmt.Println("3. check which stock exist")
	fmt.Println("4. initialize schema")
}

func main() {
	log.Println("Connecting to database")
	if err := db.Init("./data/app.db"); err != nil {
		log.Fatal(err)
	}
	defer db.DB.Close()
	log.Println("DB connected")

	for {
		printMenu()
		fmt.Print("Option: ")
		var input int
		_, err := fmt.Scan(&input)
		if err != nil {
			fmt.Println("Invalid input. Try again.")
			var discard string
			fmt.Scanln(&discard)
			continue
		}

		if input == 0 {
			fmt.Println("Exiting...")
			break
		}

		switch input {
		case 1:
			printline()
			var symbol, from, to string
			fmt.Print("Stock symbol: ")
			fmt.Scan(&symbol)
			fmt.Print("From (YYYY-MM-DD): ")
			fmt.Scan(&from)
			fmt.Print("To   (YYYY-MM-DD): ")
			fmt.Scan(&to)
			if err := db.GetStockInfo(strings.ToUpper(symbol), from, to); err != nil {
				fmt.Println("Error:", err)
			}
			printline()

		case 2:
			printline()
			var symbol string
			fmt.Print("Stock symbol: ")
			fmt.Scan(&symbol)
			symbol = strings.ToUpper(symbol)

			exists, err := db.SymbolExists(symbol)
			if err != nil {
				fmt.Println("Error:", err)
				continue
			}
			if exists {
				fmt.Printf("%s already exists in DB. Re-fetch? (y/n): ", symbol)
				var confirm string
				fmt.Scan(&confirm)
				if confirm != "y" {
					fmt.Println("Cancelled.")
					continue
				}
			}

			fmt.Printf("Fetching %s...\n", symbol)
			if err := fetchAndInsert(symbol); err != nil {
				fmt.Println("Error:", err)
				continue
			}
			fmt.Printf("%s inserted successfully.\n", symbol)
			printline()

		case 3:
			printline()
			symbols, err := db.GetSymbols()
			if err != nil {
				fmt.Println("Error:", err)
				continue
			}
			if len(symbols) == 0 {
				fmt.Println("No stocks in DB.")
			} else {
				fmt.Println("Stocks in DB:")
				for _, s := range symbols {
					fmt.Println(" -", s)
				}
			}
			printline()

		case 4:
			printline()
			if err := db.InitSchemaUser(); err != nil { //edit schema function name
				fmt.Println("Error:", err)
				continue
			}
			fmt.Println("Schema initialized.")
			printline()

		default:
			fmt.Println("Out of option. Try again.")
		}
	}
	log.Println("done")
}