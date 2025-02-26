from datetime import datetime, timedelta
import yt_dlp
from typing import List
import requests
from langgraph.prebuilt import create_react_agent
from state import State
from utils import cosine_similarity, get_openai_embedding, scrape_with_webbase
from llms import search_llm
from langchain_core.messages import SystemMessage, HumanMessage
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
import os
import time
import random
import glob


# Getting youtube video audio
def get_youtube_audio(video_url: str) -> str:
    # Downloading the audio of the video
    audio_file_name = f'audio_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    audio_ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_file_name,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True  # Suppress logs
    }

    with yt_dlp.YoutubeDL(audio_ydl_opts) as ydl:
        ydl.download([video_url])

    return audio_file_name


# Youtube channel search
def get_youtube_videos(channel_url) -> dict:
    """Fetches video links from a YouTube channel using yt_dlp."""
    ydl_opts = {
        "quiet": True,
        "extract_flat": False,  # Don't download, just extract metadata
        "skip_download": True,
        "cookiefile": 'cookies.txt',
        "no_warnings": True,
        "verbose": True,
        "playlistend": 1,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        upload_date = datetime.strptime(info['entries'][0]['upload_date'], "%Y%m%d")
        seven_days_ago = datetime.now() - timedelta(days=8)
        if upload_date > seven_days_ago:
            return {"title": info['entries'][0]["title"], "url": info['entries'][0]["webpage_url"]}
        else:
            return None


# Video search function used to pull top youtube links
def video_search(query: str, api_key, count=10, country="US", lang="en") -> List[dict]:
    """Takes in a query and returns a url of a youtube video"""
    url = "https://api.search.brave.com/res/v1/videos/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": count,
        "country": country,
        "search_lang": lang,
        "spellcheck": 1,
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        relevant_videos: List[dict] = []
        for video in result['results']:
            # Gathering video values
            selected_video: str = video['url']
            selected_title: str = video['title']
            selected_description: str = video['description']
            selected_age: str = datetime.strptime(video['page_age'], "%Y-%m-%dT%H:%M:%S")
            selected_length: str = None
            try:
                selected_length = video['video']['duration']
            except Exception:
                selected_length = None
            selected_date = datetime.strptime(video['page_age'], "%Y-%m-%dT%H:%M:%S")
            one_week_ago = datetime.now() - timedelta(days=7)
            stripped_query = query.replace(" news", "").replace("site:youtube.com ", "")

            # Generating embeddings
            query_embedding = get_openai_embedding(stripped_query)
            title_embedding = get_openai_embedding(selected_title)
            desc_embedding = get_openai_embedding(selected_description)
            title_similarity = cosine_similarity(query_embedding, title_embedding)
            desc_similarity = cosine_similarity(query_embedding, desc_embedding)

            # Checking time
            isTimeValid: bool = False
            if selected_length is not None:
                parts = list(map(int, selected_length.split(":")))
                if len(parts) == 3:  # Format: HH:MM:SS
                    h, m, s = parts
                    seconds = timedelta(hours=h, minutes=m, seconds=s).total_seconds()
                    if seconds > 120 and seconds <= 1500:
                        isTimeValid = True
                elif len(parts) == 2:  # Format: MM:SS
                    h, m, s = 0, parts[0], parts[1]
                    seconds = timedelta(hours=h, minutes=m, seconds=s).total_seconds()
                    if seconds > 180 and seconds <= 1500:
                        isTimeValid = True

            # Checking age
            seven_days_ago = datetime.now() - timedelta(days=7)
            isRecent = selected_age > seven_days_ago
                     
            # Checking relevency
            if (stripped_query in selected_title or stripped_query in selected_description) and (selected_date > one_week_ago) and ('https://www.youtube.com' in selected_video) and (selected_length is not None and isTimeValid) and (isRecent):
                relevant_videos.append({
                    "title": selected_title,
                    "url": selected_video,
                })
            elif (title_similarity > 0.78 or desc_similarity > 0.78) and (selected_date > one_week_ago) and ('https://www.youtube.com' in selected_video) and (selected_length is not None and isTimeValid) and (isRecent): 
                relevant_videos.append({
                    "title": selected_title,
                    "url": selected_video,
                })
        if len(relevant_videos) > 0:
            return relevant_videos
        else:
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    

# youtube agent
youtube_agent = create_react_agent(
    model=search_llm,
    tools=[],
    state_modifier=SystemMessage(content="""
    Your function is to summarize youtube videos. You will be given the
    the title of a video and a transcription of the video.
                                 
    Use both of these inputs to generate a two paragraph summary of the video. 
    Make sure you highlight the key points. Return your summary as a string. Include 
    as much detail as possible.
    """)
)


# Defining youtube node
def youtube_node(state: State):
    """
    YouTube search node responsible for gathering exactly 4 links and summaries for youtube videos.
    """
    print('-----Running Youtube Node-----')
    # Pulling state locally
    local_state = state.model_copy()

    # Creating deepgram client for transcribing logs
    deepgram: DeepgramClient = DeepgramClient(api_key=os.getenv('DEEPGRAM_API_KEY'))
    
    # Looping through the topics and performing video search query 
    for i in range(len(local_state.youtube_queries)):
        print('Performing youtube search for: ', local_state.desired_subjects[i])
        # Determining if this this is a youtube channel
        scrape_result = scrape_with_webbase(f'https://www.youtube.com/@{local_state.desired_subjects[i]}/videos')
        if scrape_result.strip() != '404 Not Found' and scrape_result is not None:
            # This is a youtube profile. Grabbing the most recent video
            print('Pulling video from youtube profile')
            chosen_video = get_youtube_videos(f'https://www.youtube.com/@{local_state.desired_subjects[i]}/videos')

            if chosen_video:
                print('Video found.')
                # Pulling the chosen video's audio
                audio_file_name = get_youtube_audio(chosen_video['url'])

                # Writing file to a buffer
                with open(f'{audio_file_name}.mp3', "rb") as file:
                    buffer_data = file.read()
                payload: FileSource = {
                    "buffer": buffer_data,
                }

                # Transcribing video into text
                print('Transcribing video')
                with open(f'{audio_file_name}.mp3', "rb"):
                    options = PrerecordedOptions(
                        model="nova-3",
                        smart_format=True
                    )
                transcribe_response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
                transcript = transcribe_response.results.channels[0].alternatives[0].transcript

                # Summarizing transcript
                print('Summarizing video')
                local_state.messages = [(HumanMessage(content=f"The video title: {chosen_video['title']} | The video transcript {transcript}"))]
                result = youtube_agent.invoke(local_state)
                new_message = result['messages'][-1].content

                # Adding these to state
                if chosen_video['url'] and new_message:
                    local_state.youtube_links.append(chosen_video['url'])
                    local_state.youtube_summaries.append(new_message)
                    continue
                else:
                    print('Unable to transcribe video or no video meets criteria. Passing to next topic.')
                    continue


        # Querying video search
        print('Pulling generic video')
        chosen_video: dict = None
        time.sleep(1)
        video_urls = video_search(local_state.youtube_queries[i], api_key=os.getenv("BRAVE_API_KEY"))

        # Checking result of video_url and retrying if failed
        if video_urls is None:
            print('Retrying generic video search.')
            time.sleep(1)
            video_urls = video_search(local_state.web_queries[i] + ' news', api_key=os.getenv("BRAVE_API_KEY"), count=20)
            if video_urls is None:
                print('No generic video found. Passing to next topic.')
                continue
            else:
                chosen_video = random.choice(video_urls)

        # Checking result of video_url and choosing a random video
        else:
            chosen_video = random.choice(video_urls)

        if chosen_video:
            print('Generic video found. Transcribing Audio.')
            # Pulling the chosen video's audio
            audio_file_name = get_youtube_audio(chosen_video['url'])

            # Writing file to a buffer
            with open(f'{audio_file_name}.mp3', "rb") as file:
                buffer_data = file.read()
            payload: FileSource = {
                "buffer": buffer_data,
            }

            # Transcribing video into text
            with open(f'{audio_file_name}.mp3', "rb"):
                options = PrerecordedOptions(
                    model="nova-3",
                    smart_format=True
                )
            transcribe_response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
            transcript = transcribe_response.results.channels[0].alternatives[0].transcript

            # Summarizing transcript
            print('Summarizing generic video')
            local_state.messages = [(HumanMessage(content=f"The video title: {chosen_video['title']} | The video transcript {transcript}"))]
            result = youtube_agent.invoke(local_state)
            new_message = result['messages'][-1].content

            # Adding these to state
            if chosen_video['url'] and new_message:
                local_state.youtube_links.append(chosen_video['url'])
                local_state.youtube_summaries.append(new_message) 
    
    # Updating next in local state
    if (len(local_state.youtube_links) > 0 and len(local_state.youtube_summaries) > 0 and len(local_state.youtube_links) == len(local_state.youtube_summaries)):
        local_state.next = "write"
    else:
        local_state.next = "__end__"

    # Deleting local audio files
    print('Deleting local audio files.')
    files_to_delete = glob.glob(os.path.join(".", "audio_*"))  # Find matching files
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    return local_state