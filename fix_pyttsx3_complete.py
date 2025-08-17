#!/usr/bin/env python3
"""
Complete fix for pyttsx3 macOS driver issue
"""

import os
import site

# Find the pyttsx3 driver file
site_packages = site.getsitepackages()[0]
driver_file = os.path.join(site_packages, 'pyttsx3', 'drivers', 'nsss.py')

if os.path.exists(driver_file):
    # Read the file
    with open(driver_file, 'r') as f:
        content = f.read()
    
    # Fix the imports and super() call
    fixed_content = """import objc
from Foundation import *
from AppKit import NSSpeechSynthesizer
from PyObjCTools import AppHelper
from ..voice import Voice


def buildDriver(proxy):
    return NSSpeechDriver.alloc().initWithProxy(proxy)


class NSSpeechDriver(NSObject):
    @objc.python_method
    def initWithProxy(self, proxy):
        self = objc.super(NSSpeechDriver, self).init()
        if self:
            self._proxy = proxy
            self._tts = NSSpeechSynthesizer.alloc().initWithVoice_(None)
            self._tts.setDelegate_(self)
            # default rate
            self._tts.setRate_(200)
            self._completed = True
        return self

    @objc.python_method
    def destroy(self):
        self._tts.setDelegate_(None)
        del self._tts

    @objc.python_method
    def onPumpFirst_(self, timer):
        self._proxy.setBusy(False)

    @objc.python_method
    def startLoop(self):
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.0, self, 'onPumpFirst:', None, False)
        AppHelper.runConsoleEventLoop()

    @objc.python_method
    def endLoop(self):
        AppHelper.stopEventLoop()

    @objc.python_method
    def iterate(self):
        self._proxy.setBusy(False)
        yield

    @objc.python_method
    def say(self, text):
        self._proxy.setBusy(True)
        self._proxy.notify('started-utterance')
        self._completed = True
        self._tts.startSpeakingString_(str(text))

    @objc.python_method
    def stop(self):
        if self._proxy.isBusy():
            self._completed = False
        self._tts.stopSpeaking()

    @objc.python_method
    def _toVoice(self, attr):
        try:
            lang = attr['VoiceLocaleIdentifier']
        except KeyError:
            lang = attr['VoiceLanguage']
        return Voice(attr['VoiceIdentifier'], attr['VoiceName'],
                     [lang], attr['VoiceGender'],
                     attr['VoiceAge'])

    @objc.python_method
    def getProperty(self, name):
        if name == 'voices':
            return [self._toVoice(NSSpeechSynthesizer.attributesForVoice_(v))
                    for v in list(NSSpeechSynthesizer.availableVoices())]
        elif name == 'voice':
            return self._tts.voice()
        elif name == 'rate':
            return self._tts.rate()
        elif name == 'volume':
            return self._tts.volume()
        elif name == 'pitch':
            print("Pitch adjustment not supported when using NSSS")
        else:
            raise KeyError('unknown property %s' % name)

    @objc.python_method
    def setProperty(self, name, value):
        if name == 'voice':
            # vol/rate gets reset, so store and restore it
            vol = self._tts.volume()
            rate = self._tts.rate()
            self._tts.setVoice_(value)
            self._tts.setRate_(rate)
            self._tts.setVolume_(vol)
        elif name == 'rate':
            self._tts.setRate_(value)
        elif name == 'volume':
            self._tts.setVolume_(value)
        elif name == 'pitch':
            print("Pitch adjustment not supported when using NSSS")
        else:
            raise KeyError('unknown property %s' % name)

    @objc.python_method
    def save_to_file(self, text, filename):
        self._proxy.setBusy(True)
        self._proxy.notify('started-utterance')
        url = Foundation.NSURL.fileURLWithPath_(filename)
        self._tts.startSpeakingString_toURL_(str(text), url)

    def speechSynthesizer_didFinishSpeaking_(self, tts, success):
        if not self._completed:
            success = False
        else:
            success = True
        self._proxy.notify('finished-utterance', completed=success)
        self._proxy.setBusy(False)

    def speechSynthesizer_willSpeakWord_ofString_(self, tts, rng, text):
        self._proxy.notify('started-word', location=rng.location,
                           length=rng.length)

    def speechSynthesizer_willSpeakPhoneme_(self, tts, phoneme):
        self._proxy.notify('started-phoneme', code=phoneme)

    def speechSynthesizer_didEncounterErrorAtIndex_ofString_message_(self, tts,
                                                                     index, text,
                                                                     error):
        self._proxy.notify('error', index=index, exception=Exception(error))
"""
    
    # Write the fixed content
    try:
        with open(driver_file, 'w') as f:
            f.write(fixed_content)
        print(f"✅ Fixed pyttsx3 driver at: {driver_file}")
    except PermissionError:
        print(f"❌ Permission denied. Creating a local copy instead...")
        # Create a local patched version
        local_driver = "backend/pyttsx3_nsss_patched.py"
        with open(local_driver, 'w') as f:
            f.write(fixed_content)
        print(f"✅ Created patched driver at: {local_driver}")
        print("   You'll need to copy this to the original location with proper permissions")
else:
    print(f"❌ Driver file not found at: {driver_file}")

# Test if it works now
try:
    import pyttsx3
    engine = pyttsx3.init()
    print("✅ pyttsx3 is now working!")
    engine.say("JARVIS voice system initialized")
    engine.runAndWait()
except Exception as e:
    print(f"❌ Still having issues: {e}")
    print("\nTry using espeak instead:")
    print("  brew install espeak")
    print("  Then pyttsx3 will use espeak as the TTS engine")