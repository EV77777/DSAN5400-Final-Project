import praw
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_reddit_posts_by_time(client_id, client_secret, user_agent, keywords, start_date, end_date, output_file="reddit_posts_filtered.csv", max_posts=100000):
    """
    Fetch Reddit posts globally, filter by keywords, and save all results into one file by time windows.

    Parameters:
        client_id (str): Reddit API client ID.
        client_secret (str): Reddit API client secret.
        user_agent (str): Reddit API user agent.
        keywords (list): A list of keywords to filter posts.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        output_file (str): The name of the final merged output file.
        max_posts (int): Maximum number of posts to fetch.

    Returns:
        None: Saves the filtered data to a single CSV file.
    """
    try:
        print("Initializing Reddit API client...")
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        print("Reddit API client initialized successfully.")

        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        time_window = timedelta(days=90)  # 3 months window

        all_posts = []  # To store all fetched posts
        total_fetched = 0

        while start_date_obj < end_date_obj:
            current_start = int(start_date_obj.timestamp())
            current_end = int((start_date_obj + time_window).timestamp())

            for keyword in keywords:
                print(f"Fetching posts for keyword: {keyword} from {start_date_obj} to {start_date_obj + time_window}...")

                search_results = reddit.subreddit("all").search(
                    query=keyword,
                    sort="new",
                    syntax="lucene",
                    limit=None,  # Fetch all posts in the time range
                    params={
                        "after": current_start,
                        "before": current_end,
                    }
                )

                for submission in search_results:
                    if total_fetched >= max_posts:
                        print("Reached maximum posts limit.")
                        break

                    all_posts.append({
                        "title": submission.title,
                        "subreddit": submission.subreddit.display_name,
                        "created_utc": datetime.utcfromtimestamp(submission.created_utc),
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "author": submission.author.name if submission.author else None,
                        "url": submission.url,
                        "keyword": keyword
                    })

                    total_fetched += 1

                time.sleep(1)  # Respect API rate limits

            start_date_obj += time_window
            print(f"Completed fetching posts for time window: {start_date_obj} to {start_date_obj + time_window}.")

        # Save all posts to a single CSV file
        if all_posts:
            print(f"Saving all fetched posts to {output_file}...")
            pd.DataFrame(all_posts).to_csv(output_file, index=False)
            print(f"Data saved successfully to {output_file}. Total posts: {len(all_posts)}")
        else:
            print("No posts fetched for any keyword.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define Reddit API credentials
    CLIENT_ID = "fu7JdOOMNgO8t3CbGwh5dw"
    CLIENT_SECRET = "dR9hrzvccIuKRFVKdXD-RNv-mOOf3Q"
    USER_AGENT = "5400_final_project by Sweet_Length_4114"

    # Define company-related keywords
    KEYWORDS = ["Apple", "AAPL", "Microsoft", "MSFT", "Amazon", "AMZN", "Alphabet", "GOOGL", "GOOG", "Meta", "META"]

    # Define date range
    START_DATE = "2023-01-01"
    END_DATE = "2024-01-01"

    # Fetch posts and save results
    OUTPUT_FILE = "reddit_posts_filtered.csv"
    print("Starting Reddit data fetch script...")
    fetch_reddit_posts_by_time(CLIENT_ID, CLIENT_SECRET, USER_AGENT, KEYWORDS, START_DATE, END_DATE, OUTPUT_FILE)
