import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_reddit_posts_pushshift(keywords, start_date, end_date, max_posts_per_keyword=10000, output_file="reddit_posts_filtered.csv"):
    """
    Fetch Reddit posts using Pushshift API, filter by keywords, and save results to a CSV file.

    Parameters:
        keywords (list): A list of keywords to filter posts.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        max_posts_per_keyword (int): Maximum number of posts to fetch per keyword.
        output_file (str): Name of the output CSV file.

    Returns:
        None: Saves filtered posts to a CSV file.
    """
    base_url = "https://api.pushshift.io/reddit/search/submission/"
    all_posts = []

    # Convert date strings to timestamps
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    for keyword in keywords:
        print(f"Fetching posts for keyword: {keyword}")

        params = {
            "q": keyword,
            "after": start_timestamp,
            "before": end_timestamp,
            "size": 100,  # Number of posts per request (max 100)
            "sort": "desc",
        }
        total_fetched = 0

        while total_fetched < max_posts_per_keyword:
            try:
                response = requests.get(base_url, params=params)
                if response.status_code != 200:
                    print(f"Error: {response.status_code} - {response.text}")
                    break

                data = response.json().get("data", [])
                if not data:
                    print("No more posts found for this keyword.")
                    break

                for post in data:
                    all_posts.append({
                        "title": post.get("title"),
                        "subreddit": post.get("subreddit"),
                        "created_utc": datetime.utcfromtimestamp(post["created_utc"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "author": post.get("author"),
                        "url": post.get("url"),
                        "keyword": keyword
                    })

                    total_fetched += 1
                    if total_fetched >= max_posts_per_keyword:
                        break

                print(f"Fetched {total_fetched} posts for keyword: {keyword} so far...")

                # Update 'before' parameter for pagination
                params["before"] = data[-1]["created_utc"]

                # Respect rate limits
                time.sleep(1)

            except Exception as e:
                print(f"An error occurred: {e}")
                break

    # Save all posts to a CSV file
    if all_posts:
        print(f"Saving all fetched posts to {output_file}...")
        pd.DataFrame(all_posts).to_csv(output_file, index=False)
        print(f"Data saved successfully to {output_file}. Total posts: {len(all_posts)}")
    else:
        print("No posts fetched for any keyword.")

if __name__ == "__main__":
    # Define keywords
    KEYWORDS = ["Apple", "AAPL", "Microsoft", "MSFT", "Amazon", "AMZN", "Alphabet", "GOOGL", "GOOG", "Meta", "META"]

    # Define date range (last 3 months)
    START_DATE = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    END_DATE = datetime.now().strftime("%Y-%m-%d")

    # Maximum posts per keyword
    MAX_POSTS_PER_KEYWORD = 20000

    # Output file
    OUTPUT_FILE = "reddit_posts_filtered_pushshift.csv"

    # Fetch posts
    print("Starting Reddit data fetch using Pushshift API...")
    fetch_reddit_posts_pushshift(KEYWORDS, START_DATE, END_DATE, MAX_POSTS_PER_KEYWORD, OUTPUT_FILE)

