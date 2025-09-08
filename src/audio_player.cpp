#include "audio_player.hpp"
#include <AL/al.h>
#include <AL/alc.h>

AudioPlayer::AudioPlayer() 
    :   device(nullptr),
        context(nullptr),
        source(0),
        buffer(0)
{}

void AudioPlayer::play() {
    alSourcePlay(source);

    if (alGetError() != AL_NO_ERROR) { \
        printf("AudioPLayer: audio broke\n");
    }
}

bool AudioPlayer::init() {
    device = alcOpenDevice(nullptr);
    if (!device) {
        return false;
    }

    context = alcCreateContext(device, nullptr);
    alcMakeContextCurrent(context);

    alGenSources(1, &source);
    if (alGetError() != AL_NO_ERROR) { \
        printf("AudioPLayer: couldn't generate source\n");
        return false;
    }
    alGenBuffers(1, &buffer);
    if (alGetError() != AL_NO_ERROR) { \
        printf("AudioPLayer: couldn't generate buffer\n");
        return false;
    }

    return true;
}

bool AudioPlayer::loadMP3(const char* filename) {
    mpg123_handle* mh;
    unsigned char* tmpBuffer;
    size_t bufferSize;
    size_t done;
    int err;

    mpg123_init();
    mh = mpg123_new(nullptr, &err);
    bufferSize = mpg123_outblock(mh);
    tmpBuffer = new unsigned char[bufferSize];

    if (mpg123_open(mh, filename) != MPG123_OK ||
        mpg123_format_all(mh) != MPG123_OK) {
        delete[] tmpBuffer;
        mpg123_delete(mh);
        return false;
    }

    long rate;
    int channels, encoding;
    mpg123_getformat(mh, &rate, &channels, &encoding);

    std::vector<unsigned char> pcm_data;
    int readResult;

    while ((readResult = mpg123_read(mh, tmpBuffer, bufferSize, &done)) == MPG123_OK) {
        pcm_data.insert(pcm_data.end(), tmpBuffer, tmpBuffer + done);
    }

    ALenum format = (channels == 2) ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16;

    alBufferData(buffer, format, pcm_data.data(), pcm_data.size(), rate);
    if (alGetError() != AL_NO_ERROR) { \
        printf("AudioPLayer: couldn't generate buffer\n");
        return false;
    }

    delete[] tmpBuffer;
    mpg123_close(mh);
    mpg123_delete(mh);
    mpg123_exit();
    
    alSourcei(source, AL_BUFFER, buffer);
    if (alGetError() != AL_NO_ERROR) { \
        printf("AudioPLayer: couldn't generate buffer\n");
        return false;
    }

    return true;
}

void AudioPlayer::cleanup() {
    alDeleteSources(1, &source);
    alDeleteBuffers(1, &buffer);
    if (alGetError() != AL_NO_ERROR) { \
        printf("AudioPLayer: cleanup failed\n");
    }

    alcMakeContextCurrent(nullptr);
    if (context) {
        alcDestroyContext(context);
    }
    if (device) {
        alcCloseDevice(device);
    }
}