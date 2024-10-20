from model_inference import generate_response

def main():

    query = input("Enter your query: ")
    response = generate_response(query)
    
    print("Generated Response:", response)

if __name__ == "__main__":
    main()