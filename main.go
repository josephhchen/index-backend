package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/auth"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
)

type Product struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Category    string `json:"category"`
}

type SearchRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type SearchResponse struct {
	Products []Product `json:"products"`
	Count    int       `json:"count"`
}

var client *weaviate.Client

func initWeaviate() {
	cfg := weaviate.Config{
		Host:   getEnv("WEAVIATE_HOST", "localhost:8080"),
		Scheme: "http",
	}

	if apiKey := os.Getenv("WEAVIATE_API_KEY"); apiKey != "" {
		cfg.AuthConfig = auth.ApiKey{Value: apiKey}
	}

	client = weaviate.New(cfg)

	createSchema()
	
	loadProducts()
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func createSchema() {
	className := "Product"
	
	exists, err := client.Schema().ClassExistenceChecker().WithClassName(className).Do(context.Background())
	if err != nil {
		log.Printf("Error checking class existence: %v", err)
		return
	}
	
	if exists {
		log.Printf("Class %s already exists", className)
		return
	}

	classObj := &models.Class{
		Class: className,
		Properties: []*models.Property{
			{
				Name:     "name",
				DataType: []string{"text"},
			},
			{
				Name:     "description",
				DataType: []string{"text"},
			},
			{
				Name:     "category",
				DataType: []string{"text"},
			},
		},
		Vectorizer: "text2vec-openai",
	}

	err = client.Schema().ClassCreator().WithClass(classObj).Do(context.Background())
	if err != nil {
		log.Printf("Error creating schema: %v", err)
	} else {
		log.Printf("Schema created successfully")
	}
}

func loadProducts() {
	result, err := client.GraphQL().Aggregate().WithClassName("Product").WithFields(graphql.Field{Name: "meta", Fields: []graphql.Field{{Name: "count"}}}).Do(context.Background())
	if err == nil {
		if data, ok := result.Data["Aggregate"].(map[string]interface{}); ok {
			if products, ok := data["Product"].([]interface{}); ok && len(products) > 0 {
				if product, ok := products[0].(map[string]interface{}); ok {
					if meta, ok := product["meta"].(map[string]interface{}); ok {
						if count, ok := meta["count"].(float64); ok && count > 0 {
							log.Printf("Products already loaded: %v", count)
							return
						}
					}
				}
			}
		}
	}

	file, err := os.Open("documents.txt")
	if err != nil {
		log.Printf("Error opening documents.txt: %v", err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	products := []Product{}
	id := 1

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, " - ", 2)
		if len(parts) != 2 {
			continue
		}

		name := parts[0]
		description := parts[1]
		category := categorizeProduct(name, description)

		product := Product{
			ID:          strconv.Itoa(id),
			Name:        name,
			Description: description,
			Category:    category,
		}

		products = append(products, product)
		id++
	}

	batcher := client.Batch().ObjectsBatcher()
	for _, product := range products {
		obj := &models.Object{
			Class: "Product",
			Properties: map[string]interface{}{
				"name":        product.Name,
				"description": product.Description,
				"category":    product.Category,
			},
		}
		batcher = batcher.WithObject(obj)
	}

	_, err = batcher.Do(context.Background())
	if err != nil {
		log.Printf("Error batch inserting products: %v", err)
	} else {
		log.Printf("Successfully loaded %d products", len(products))
	}
}

// Predefined categories for consistent classification
var productCategories = []string{
	"smartphones", "laptops", "tablets", "audio", "wearables",
	"cameras", "gaming", "automotive", "appliances", "fitness",
	"e-readers", "smart-home", "accessories", "electronics",
}

// OpenAI API structures
type OpenAIRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	MaxTokens   int       `json:"max_tokens"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIResponse struct {
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Message Message `json:"message"`
}

// AI-powered categorization using OpenAI
func categorizeProductAI(name, description string) string {
	// Create prompt for categorization
	prompt := fmt.Sprintf(`Categorize this product into one of these categories: %s

Product: %s
Description: %s

Return only the category name that best fits this product. Choose the most specific and appropriate category.`,
		strings.Join(productCategories, ", "), name, description)

	reqBody := OpenAIRequest{
		Model: "gpt-3.5-turbo",
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
		Temperature: 0.1,
		MaxTokens:   50,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		log.Printf("Error marshaling OpenAI request: %v", err)
		return categorizeProductFallback(name)
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Error creating OpenAI request: %v", err)
		return categorizeProductFallback(name)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_KEY"))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error calling OpenAI API: %v", err)
		return categorizeProductFallback(name)
	}
	defer resp.Body.Close()

	var openaiResp OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		log.Printf("Error decoding OpenAI response: %v", err)
		return categorizeProductFallback(name)
	}

	if len(openaiResp.Choices) > 0 {
		category := strings.ToLower(strings.TrimSpace(openaiResp.Choices[0].Message.Content))
		
		for _, validCategory := range productCategories {
			if category == validCategory {
				return category
			}
		}
	}

	return categorizeProductFallback(name)
}

func categorizeProductFallback(name string) string {
	name = strings.ToLower(name)
	
	categories := map[string][]string{
		"smartphones": {"phone", "iphone", "galaxy", "pixel", "oneplus", "samsung", "mobile"},
		"laptops":     {"laptop", "macbook", "thinkpad", "surface pro", "notebook", "chromebook"},
		"tablets":     {"ipad", "tablet", "surface", "kindle fire"},
		"audio":       {"airpods", "headphones", "earbuds", "speaker", "soundbar", "audio", "beats", "sony wh"},
		"wearables":   {"watch", "fitbit", "band", "tracker", "smartwatch", "apple watch"},
		"cameras":     {"camera", "gopro", "canon", "nikon", "photography", "lens"},
		"gaming":      {"console", "nintendo", "playstation", "xbox", "gaming", "switch"},
		"automotive":  {"tesla", "car", "vehicle", "auto"},
		"appliances":  {"vacuum", "roomba", "dyson", "cleaner", "kitchenaid", "mixer"},
		"fitness":     {"bike", "peloton", "treadmill", "exercise", "workout", "fitness"},
		"e-readers":   {"kindle", "e-reader", "ebook"},
		"smart-home":  {"alexa", "echo", "nest", "smart", "home"},
	}

	for category, keywords := range categories {
		for _, keyword := range keywords {
			if strings.Contains(name, keyword) {
				return category
			}
		}
	}
	
	return "electronics"
}

func categorizeProduct(name, description string) string {
	// For production: use AI categorization
	// For development: uncomment the line below to use AI
	// return categorizeProductAI(name, description)
	
	return categorizeProductFallback(name)
}

func searchProducts(c *gin.Context) {
	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Limit == 0 {
		req.Limit = 10
	}

	nearText := client.GraphQL().NearTextArgBuilder().
		WithConcepts([]string{req.Query})
	
	result, err := client.GraphQL().Get().
		WithClassName("Product").
		WithFields(graphql.Field{Name: "name"}, graphql.Field{Name: "description"}, graphql.Field{Name: "category"}).
		WithNearText(nearText).
		WithLimit(req.Limit).
		Do(context.Background())

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	products := []Product{}
	if data, ok := result.Data["Get"].(map[string]interface{}); ok {
		if productData, ok := data["Product"].([]interface{}); ok {
			for i, item := range productData {
				if productMap, ok := item.(map[string]interface{}); ok {
					product := Product{
						ID:          strconv.Itoa(i + 1),
						Name:        getString(productMap, "name"),
						Description: getString(productMap, "description"),
						Category:    getString(productMap, "category"),
					}
					products = append(products, product)
				}
			}
		}
	}

	response := SearchResponse{
		Products: products,
		Count:    len(products),
	}

	c.JSON(http.StatusOK, response)
}

func getRecommendations(c *gin.Context) {
	productName := c.Query("product")
	if productName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "product parameter is required"})
		return
	}

	limit := 5
	if l := c.Query("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil {
			limit = parsed
		}
	}

	nearText := client.GraphQL().NearTextArgBuilder().
		WithConcepts([]string{productName})
	
	result, err := client.GraphQL().Get().
		WithClassName("Product").
		WithFields(graphql.Field{Name: "name"}, graphql.Field{Name: "description"}, graphql.Field{Name: "category"}).
		WithNearText(nearText).
		WithLimit(limit).
		Do(context.Background())

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	products := []Product{}
	if data, ok := result.Data["Get"].(map[string]interface{}); ok {
		if productData, ok := data["Product"].([]interface{}); ok {
			for i, item := range productData {
				if productMap, ok := item.(map[string]interface{}); ok {
					product := Product{
						ID:          strconv.Itoa(i + 1),
						Name:        getString(productMap, "name"),
						Description: getString(productMap, "description"),
						Category:    getString(productMap, "category"),
					}
					products = append(products, product)
				}
			}
		}
	}

	response := SearchResponse{
		Products: products,
		Count:    len(products),
	}

	c.JSON(http.StatusOK, response)
}

func getString(m map[string]interface{}, key string) string {
	if val, ok := m[key]; ok {
		if str, ok := val.(string); ok {
			return str
		}
	}
	return ""
}

func healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "healthy",
		"service": "vector-search-api",
	})
}

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found")
	}

	initWeaviate()

	r := gin.Default()

	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		AllowCredentials: true,
	}))

	r.GET("/health", healthCheck)
	r.POST("/search", searchProducts)
	r.GET("/recommendations", getRecommendations)

	port := getEnv("PORT", "8080")
	fmt.Printf("Server starting on port %s\n", port)
	log.Fatal(r.Run(":" + port))
}