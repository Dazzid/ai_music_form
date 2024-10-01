import os
import argparse
from tqdm import tqdm
import formExtractor as fem

def load_audio_files(path):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mp3"):
                audio_files.append(os.path.join(root, file))
    return audio_files

def main(source):
    base_path = f'/home/laura/aimir/{source}/'
    save_to_path = os.path.join(base_path, 'form/')
    
    song_files = os.path.join(base_path, 'audio')
    files = load_audio_files(song_files)
    print('\n-----------------------------\n')
    print('number of files:', len(files))
    print('-----------------------------\n')
    print('Processing files...')
    form_data = fem.formExtractor()

    failed_files = []
    K = 4

    for song in tqdm(files):
        id_file = os.path.basename(song).split('.')[0]
        try:
            form_data.getFormAndSave(K, song, id_file, save_to_path)
        except Exception as e:
            print(f'Failed to process {id_file}.mp3: {e}')
            failed_files.append(song)

    print('Process Completed!')
    if failed_files:
        print('Files that could not be processed:', failed_files)
        #save files into a txt
        filePath = os.path.join(save_to_path, '00_failed_files.txt')
        with open(filePath, 'w') as f:
            for item in failed_files:
                f.write("%s\n" % item)
        print(f'Failed files saved to {filePath}')
    else:
        print('All files were processed successfully!')

if __name__ == "__main__":
    class CustomArgParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            print(f'\nError: {message}')
            print('\nAllowed values for source are: suno, lastfm, boomy')
            self.exit(2)
    
    parser = CustomArgParser(description='Process audio files from specified source.')
    parser.add_argument('source', choices=['suno', 'lastfm', 'boomy'], help='The source of the audio files to process (suno, lastfm, or boomy)')
    args = parser.parse_args()
    
    main(args.source)
