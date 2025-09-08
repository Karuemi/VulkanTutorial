#ifndef AUDIO_PLAYER_HPP
#define AUDIO_PLAYER_HPP

#include <AL/al.h>
#include <AL/alc.h>
#include <mpg123.h>

#include <stdio.h>
#include <vector>

class AudioPlayer {
public:
    AudioPlayer();

    void play();
    bool init();
    bool loadMP3(const char* filename);
    void cleanup();

private:
    ALCdevice* device;
    ALCcontext* context;
    ALuint source;
    ALuint buffer;
};

#endif