'''
    Preprocess data CSV so it only contains the columns of interest for the task.

    Currently developed so it works either by selecting an individual audio and video source 
    for analysis and capbility to select multiple bio sources. Alternatively, if none of these 
    are set, none of the columns for audio, video and bio data will be removed.

    If deemed in the future, can develop so can select multiple audio or video sources or 
    filter based on modality.
'''

import pandas as pd


def preprocess_data(data_in_path, data_out_path, audio_source=None, video_source=None, bio_sources=None, individual_model=False):
    '''
        Preprocess data to only contain the columns of interest for the task.

        Args:
            data_in_path (str): path to the data CSV
            data_out_path (str): path to save the processed data CSV
            audio_source (str): source of the audio data (phone, computer, mic)
            video_source (str): source of the video data (phone, computer)
            bio_sources (list): list of sources of the bio data (pzt, emg, eeg_1, eeg_2)

        Returns:
            data (pd.DataFrame): dataframe with selected columns for the task
    '''

    data = pd.read_csv(data_in_path)
    # make sure everything is in lower case for comparisons
    data.columns = map(str.lower, data.columns)

    if individual_model:
        data = clean_metadata_for_individual_model(data)
    if audio_source:
        data = select_model_audio_source(data, audio_source)
    if video_source:
        data = select_model_video_source(data, video_source)
    if bio_sources:
        data = select_model_biodata(data, bio_sources)

    data = clean_null_data(data)
    data = clean_unidentified_pitches(data, audio_source)

    data.to_csv(data_out_path, index=False)
    

def clean_metadata_for_individual_model(data):
    '''
        When building individual model, can disregard participant number, age, sex, 
        expereince level from metadata. This should make processing a bit faster.

        Args:
            data (pd.DataFrame): dataframe of all data

        Returns:
            clean_data (pd.DataFrame): dataframe with selected metadata columns removed
    '''
    clean_data = data.drop(columns=['participant_number', 'sex', 'age', 'experience_level'])
    return clean_data


def select_model_audio_source(data, selected_source):
    '''
        Given a source, keep the columns that correspond to that source for the audio.

        Args:
            data (pd.DataFrame): dataframe of all data
            selected_source (str): source of the data (phone, computer, mic_computer)

        Returns:
            data (pd.DataFrame): dataframe with selected audio columns for the source
    '''

    audio_sources = ['mic', 'phone', 'computer']
    columns_to_drop = []

    for audio_source in audio_sources:
        if audio_source != selected_source.lower():
            columns_to_drop.extend([
                f'{audio_source}_pitch', f'{audio_source}_note', f'{audio_source}_rms_energy', f'{audio_source}_spec_cent', 
                f'{audio_source}_spec_spread', f'{audio_source}_spec_skew', f'{audio_source}_spec_kurt', 
                f'{audio_source}_spec_slope', f'{audio_source}_spec_decr', f'{audio_source}_spec_rolloff',
                f'{audio_source}_spec_flat', f'{audio_source}_spec_crest'
            ])
            columns_to_drop.extend([f'{audio_source}_tristimulus{i}' for i in range(1, 4)])
            columns_to_drop.extend([f'{audio_source}_mfcc_{i}' for i in range(1, 14)])

    data = data.drop(columns=columns_to_drop)

    return data
    

def select_model_video_source(data, selected_source):
    '''
        Given a source, keep the columns that correspond to that source for the video.

        Args:
            data (pd.DataFrame): dataframe of all data
            source (str): source of the data (phone, computer, mic_computer)

        Returns:
            data (pd.DataFrame): dataframe with selected video columns for the source
    '''
    video_sources = [
        'phone', 
        'computer'
    ]
    columns_to_drop = []

    for video_source in video_sources:
        if video_source != selected_source.lower():
            columns_to_drop.extend([f'{video_source}_pose_landmark_{i}_x' for i in range(1, 34)])
            columns_to_drop.extend([f'{video_source}_pose_landmark_{i}_y' for i in range(1, 34)])
            columns_to_drop.extend([f'{video_source}_pose_landmark_{i}_z' for i in range(1, 34)])
            columns_to_drop.extend([f'{video_source}_face_landmark_{i}_x' for i in range(1, 69)])
            columns_to_drop.extend([f'{video_source}_face_landmark_{i}_y' for i in range(1, 69)])
            columns_to_drop.extend([f'{video_source}_face_landmark_{i}_z' for i in range(1, 69)])
            break

    data = data.drop(columns=columns_to_drop)
    
    return data


def select_model_biodata(data, selected_sources):
    '''
        Given a source, keep the columns that correspond to that source for the bio data.

        Args:
            data (pd.DataFrame): dataframe of all data
            sources (list): list of sources of the data (PZT, EMG, EEG_1, EEG_2)

        Returns:
            data (pd.DataFrame): dataframe with selected bio columns for the source
    '''
    bio_sources = [
        'respiration_1',
        'emg_1',
        'eeg_1', 
        'eeg_2'
    ]
    columns_to_drop = []

    selected_sources_lower = [source.lower() for source in selected_sources]

    # Check each bio source against selected sources
    for bio_source in bio_sources:
        bio_source_lower = bio_source.lower()
        if all(bio_source_lower not in source for source in selected_sources_lower):
            columns_to_drop.extend([col for col in data.columns if col.lower().startswith(bio_source_lower)])

    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    return data


def clean_null_data(data):
    '''
        Remove columns with more than 50% of the data missing.

        Args:
            data (pd.DataFrame): dataframe of all data

        Returns:
            data (pd.DataFrame): dataframe with columns with more than 50% missing data removed
    '''
    data = data.dropna(axis=1, thresh=int(0.5*len(data)))
    return data


def clean_unidentified_pitches(data, source):
    '''
    Remove rows with unidentified pitches for the specified audio sources.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame.
        source (str): The audio source to filter on (mic, phone, computer).
    
    Returns:
        pd.DataFrame: The cleaned DataFrame with rows containing unidentified pitches removed.
    '''
    # Start with a condition that keeps all rows
    combined_condition = pd.Series([True] * len(data))

    if source: # only look for the one source thats already filtered
        pitch_col = f'{source}_pitch'
        note_col = f'{source}_note'
        condition = (data[pitch_col] != 0.000) | (data[note_col] != 'Rest')
        combined_condition &= condition
    else: # remove for all sources if they havent been filtered
        for source in ['mic', 'phone', 'computer']:
            pitch_col = f'{source}_pitch'
            note_col = f'{source}_note'
            condition = (data[pitch_col] != 0.000) | (data[note_col] != 'Rest')
            combined_condition &= condition

    # Filter the data based on the combined condition
    cleaned_data = data[combined_condition]

    return cleaned_data
