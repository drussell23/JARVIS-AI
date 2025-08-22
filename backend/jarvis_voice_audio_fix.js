// Add this to JarvisVoice.js to handle audio-capture errors better

      recognitionRef.current.onerror = async (event) => {
        console.error('Speech recognition error:', event.error, event);
        
        // Handle different error types
        switch(event.error) {
          case 'audio-capture':
            setError('🎤 Microphone access denied. Please grant permission and reload.');
            setMicStatus('error');
            
            // Try to request permissions again
            try {
              const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
              stream.getTracks().forEach(track => track.stop());
              
              // If we get here, permissions were granted
              setError('');
              setMicStatus('ready');
              startListening();
            } catch (e) {
              console.error('Failed to get microphone access:', e);
              
              // Show detailed instructions
              const instructions = `
                🎤 Microphone Access Required
                
                Please follow these steps:
                1. Click the lock icon 🔒 in your browser's address bar
                2. Set Microphone to "Allow"
                3. Reload the page
                
                Or check System Preferences → Security & Privacy → Microphone
              `;
              setError(instructions);
            }
            break;
            
          case 'not-allowed':
            setError('🚫 Microphone permission denied. Please enable in browser settings.');
            setMicStatus('error');
            break;
            
          case 'no-speech':
            // Silently restart for no-speech
            if (continuousListening && isListening) {
              setTimeout(() => {
                try {
                  recognitionRef.current.start();
                } catch (e) {
                  console.log('Restarting recognition...');
                }
              }, 100);
            }
            break;
            
          case 'network':
            setError('🌐 Network error. Check your connection.');
            break;
            
          default:
            if (event.error !== 'aborted') {
              setError(`Speech recognition error: ${event.error}`);
            }
        }
      };
