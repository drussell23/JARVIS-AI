/**
 * Audio helper utilities for JARVIS voice output
 */

class AudioHelper {
  constructor() {
    this.audioQueue = [];
    this.isPlaying = false;
    this.currentAudio = null;
  }

  /**
   * Play audio with enhanced error handling and debugging
   */
  async playAudio(text, apiUrl) {
    console.log('[AudioHelper] Playing audio for text:', text.substring(0, 50) + '...');
    
    try {
      // Use GET for short text, POST for long text
      const usePost = text.length > 500 || text.includes('\n');
      
      if (!usePost) {
        // GET method
        const audioUrl = `${apiUrl}/audio/speak/${encodeURIComponent(text)}`;
        console.log('[AudioHelper] Using GET method:', audioUrl);
        
        const audio = new Audio();
        
        // Set up promise-based playback
        return new Promise((resolve, reject) => {
          audio.oncanplaythrough = () => {
            console.log('[AudioHelper] Audio can play through');
          };
          
          audio.onplay = () => {
            console.log('[AudioHelper] Audio started playing');
          };
          
          audio.onended = () => {
            console.log('[AudioHelper] Audio playback completed');
            this.currentAudio = null;
            resolve();
          };
          
          audio.onerror = async (e) => {
            console.error('[AudioHelper] GET audio error:', e);
            console.log('[AudioHelper] Falling back to POST method');
            
            // Fallback to POST
            try {
              await this.playAudioPost(text, apiUrl);
              resolve();
            } catch (postError) {
              reject(postError);
            }
          };
          
          // Set properties and play
          audio.src = audioUrl;
          audio.volume = 1.0;
          this.currentAudio = audio;
          
          audio.play().catch(err => {
            console.error('[AudioHelper] Play promise rejected:', err);
            reject(err);
          });
        });
      } else {
        // POST method for long text
        return this.playAudioPost(text, apiUrl);
      }
    } catch (error) {
      console.error('[AudioHelper] Error playing audio:', error);
      throw error;
    }
  }

  /**
   * Play audio using POST method
   */
  async playAudioPost(text, apiUrl) {
    console.log('[AudioHelper] Using POST method for audio');
    
    try {
      const response = await fetch(`${apiUrl}/audio/speak`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const blob = await response.blob();
      console.log('[AudioHelper] Audio blob received:', blob.size, 'bytes');
      
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);
      
      return new Promise((resolve, reject) => {
        audio.onended = () => {
          console.log('[AudioHelper] POST audio completed');
          URL.revokeObjectURL(audioUrl);
          this.currentAudio = null;
          resolve();
        };
        
        audio.onerror = (e) => {
          console.error('[AudioHelper] POST audio error:', e);
          URL.revokeObjectURL(audioUrl);
          reject(e);
        };
        
        audio.volume = 1.0;
        this.currentAudio = audio;
        
        audio.play().catch(reject);
      });
    } catch (error) {
      console.error('[AudioHelper] POST method error:', error);
      throw error;
    }
  }

  /**
   * Queue audio for sequential playback
   */
  async queueAudio(text, apiUrl) {
    this.audioQueue.push({ text, apiUrl });
    
    if (!this.isPlaying) {
      this.processQueue();
    }
  }

  /**
   * Process audio queue
   */
  async processQueue() {
    if (this.audioQueue.length === 0) {
      this.isPlaying = false;
      return;
    }
    
    this.isPlaying = true;
    const { text, apiUrl } = this.audioQueue.shift();
    
    try {
      await this.playAudio(text, apiUrl);
    } catch (error) {
      console.error('[AudioHelper] Queue processing error:', error);
    }
    
    // Process next in queue
    this.processQueue();
  }

  /**
   * Stop current audio
   */
  stopAudio() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio = null;
    }
    this.audioQueue = [];
    this.isPlaying = false;
  }
}

// Export singleton instance
export default new AudioHelper();