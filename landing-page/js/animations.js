// JARVIS Landing Page - Advanced Animations

document.addEventListener('DOMContentLoaded', function() {
    // Arc Reactor Energy Flow
    function createEnergyParticle() {
        const reactor = document.querySelector('.arc-reactor-large');
        if (!reactor) return;
        
        const particle = document.createElement('div');
        particle.className = 'energy-particle';
        
        // Random starting position on the outer ring
        const angle = Math.random() * Math.PI * 2;
        const radius = 150; // Outer ring radius
        const startX = Math.cos(angle) * radius;
        const startY = Math.sin(angle) * radius;
        
        particle.style.left = `${150 + startX}px`;
        particle.style.top = `${150 + startY}px`;
        
        reactor.appendChild(particle);
        
        // Animate to center
        particle.animate([
            { transform: 'translate(-50%, -50%) scale(1)', opacity: 1 },
            { transform: 'translate(-50%, -50%) scale(0)', opacity: 0, offset: 0.8 },
        ], {
            duration: 2000,
            easing: 'ease-in'
        }).onfinish = () => particle.remove();
    }
    
    // Create energy particles periodically
    setInterval(createEnergyParticle, 300);
    
    // Dynamic HUD Data
    function updateHUDData() {
        const systemStatus = document.querySelector('.system-status');
        const powerLevel = document.querySelector('.power-level');
        
        if (systemStatus) {
            const statuses = ['ONLINE', 'ACTIVE', 'READY', 'OPERATIONAL'];
            systemStatus.textContent = statuses[Math.floor(Math.random() * statuses.length)];
        }
        
        if (powerLevel) {
            const level = Math.floor(Math.random() * 20) + 80; // 80-100%
            powerLevel.textContent = `${level}%`;
        }
    }
    
    setInterval(updateHUDData, 3000);
    
    // Holographic Scan Effect
    function createScanLine() {
        const features = document.querySelector('.features');
        if (!features) return;
        
        const scanLine = document.createElement('div');
        scanLine.className = 'scan-line';
        scanLine.style.cssText = `
            position: absolute;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--arc-blue), transparent);
            top: 0;
            left: 0;
            z-index: 10;
            pointer-events: none;
        `;
        
        features.appendChild(scanLine);
        
        scanLine.animate([
            { top: '0%' },
            { top: '100%' }
        ], {
            duration: 3000,
            easing: 'linear'
        }).onfinish = () => scanLine.remove();
    }
    
    setInterval(createScanLine, 5000);
    
    // Interactive Arc Reactor Response
    const arcReactor = document.querySelector('.arc-reactor-large');
    if (arcReactor) {
        document.addEventListener('mousemove', (e) => {
            const rect = arcReactor.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            
            const angleX = (e.clientY - centerY) / 50;
            const angleY = (e.clientX - centerX) / 50;
            
            arcReactor.style.transform = `perspective(1000px) rotateX(${-angleX}deg) rotateY(${angleY}deg)`;
        });
    }
    
    // Tech Card Magnetic Effect
    const techCards = document.querySelectorAll('.tech-card');
    techCards.forEach(card => {
        card.addEventListener('mouseenter', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            this.style.setProperty('--mouse-x', `${x}px`);
            this.style.setProperty('--mouse-y', `${y}px`);
            
            // Add magnetic pull effect
            const nearbyCards = Array.from(techCards).filter(c => c !== this);
            nearbyCards.forEach(nearby => {
                const nearbyRect = nearby.getBoundingClientRect();
                const distance = Math.sqrt(
                    Math.pow(rect.left - nearbyRect.left, 2) + 
                    Math.pow(rect.top - nearbyRect.top, 2)
                );
                
                if (distance < 200) {
                    const pullX = (rect.left - nearbyRect.left) / 10;
                    const pullY = (rect.top - nearbyRect.top) / 10;
                    nearby.style.transform = `translate(${pullX}px, ${pullY}px)`;
                }
            });
        });
        
        card.addEventListener('mouseleave', function() {
            techCards.forEach(c => {
                c.style.transform = '';
            });
        });
    });
    
    // Voice Command Animation
    let isListening = false;
    const voiceButton = document.querySelector('.demo-button');
    
    if (voiceButton) {
        voiceButton.addEventListener('click', function() {
            isListening = !isListening;
            
            if (isListening) {
                this.classList.add('listening');
                this.innerHTML = '<i class="fas fa-microphone-alt"></i> Listening...';
                createVoiceWaves();
            } else {
                this.classList.remove('listening');
                this.innerHTML = '<i class="fas fa-microphone"></i> Try Voice Command';
                stopVoiceWaves();
            }
        });
    }
    
    function createVoiceWaves() {
        const visualizer = document.querySelector('.voice-visualizer');
        if (!visualizer) return;
        
        visualizer.innerHTML = '';
        for (let i = 0; i < 20; i++) {
            const bar = document.createElement('div');
            bar.className = 'wave-bar';
            bar.style.setProperty('--i', i);
            visualizer.appendChild(bar);
        }
    }
    
    function stopVoiceWaves() {
        const visualizer = document.querySelector('.voice-visualizer');
        if (visualizer) {
            visualizer.innerHTML = '';
        }
    }
    
    // Capability Item Activation
    const capabilities = document.querySelectorAll('.capability-item');
    capabilities.forEach((item, index) => {
        item.addEventListener('click', function() {
            // Remove active from all
            capabilities.forEach(cap => cap.classList.remove('active'));
            
            // Add active to clicked
            this.classList.add('active');
            
            // Update hologram display
            const display = document.querySelector('.command-preview');
            if (display) {
                const commands = {
                    0: '> jarvis.execute("open_application", {"name": "Chrome"})',
                    1: '> jarvis.system.control_volume(50)',
                    2: '> jarvis.capture_screen({"area": "full", "save": true})',
                    3: '> jarvis.execute_routine("morning_productivity")'
                };
                
                display.textContent = commands[index] || '> jarvis.ready()';
            }
        });
    });
    
    // Installation Step Progress
    const stepCards = document.querySelectorAll('.step-card');
    stepCards.forEach((card, index) => {
        card.addEventListener('click', function() {
            this.classList.toggle('completed');
            
            // Check if all steps completed
            const completed = document.querySelectorAll('.step-card.completed').length;
            if (completed === stepCards.length) {
                showSuccessMessage();
            }
        });
    });
    
    function showSuccessMessage() {
        const message = document.createElement('div');
        message.className = 'success-message';
        message.innerHTML = `
            <div class="success-content">
                <i class="fas fa-check-circle"></i>
                <h3>JARVIS Initialized Successfully!</h3>
                <p>Your AI assistant is ready to serve.</p>
            </div>
        `;
        message.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 212, 255, 0.1);
            border: 2px solid var(--arc-blue);
            padding: 2rem;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            z-index: 9999;
            animation: success-pop 0.5s ease-out;
        `;
        
        document.body.appendChild(message);
        
        setTimeout(() => {
            message.style.animation = 'success-fade 0.5s ease-out forwards';
            setTimeout(() => message.remove(), 500);
        }, 3000);
    }
    
    // Add CSS for new animations
    const style = document.createElement('style');
    style.textContent = `
        .energy-particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: var(--arc-blue);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--arc-blue);
            pointer-events: none;
        }
        
        .demo-button.listening {
            animation: pulse-button 1s ease-in-out infinite;
            background: var(--arc-blue);
            color: var(--dark-bg);
        }
        
        @keyframes pulse-button {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .capability-item.active {
            background: rgba(0, 212, 255, 0.1);
            border-color: var(--arc-blue);
            transform: translateX(10px);
        }
        
        .step-card.completed {
            background: rgba(0, 255, 0, 0.05);
            border-color: rgba(0, 255, 0, 0.5);
        }
        
        .step-card.completed::after {
            content: '✓';
            position: absolute;
            top: 1rem;
            right: 1rem;
            color: #00ff00;
            font-size: 2rem;
            animation: check-mark 0.5s ease-out;
        }
        
        @keyframes check-mark {
            0% { transform: scale(0) rotate(0deg); }
            50% { transform: scale(1.2) rotate(180deg); }
            100% { transform: scale(1) rotate(360deg); }
        }
        
        @keyframes success-pop {
            0% { transform: translate(-50%, -50%) scale(0); opacity: 0; }
            50% { transform: translate(-50%, -50%) scale(1.1); }
            100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }
        
        @keyframes success-fade {
            to { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
        }
        
        .jarvis-mode {
            animation: jarvis-activate 3s ease-out;
        }
        
        @keyframes jarvis-activate {
            0% { filter: hue-rotate(0deg) brightness(1); }
            50% { filter: hue-rotate(180deg) brightness(1.5); }
            100% { filter: hue-rotate(360deg) brightness(1); }
        }
        
        .jarvis-message {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-family: var(--font-primary);
            font-size: 3rem;
            color: var(--arc-blue);
            text-shadow: 0 0 50px var(--arc-blue);
            animation: jarvis-text 3s ease-out;
            pointer-events: none;
            z-index: 10000;
        }
        
        @keyframes jarvis-text {
            0% { opacity: 0; letter-spacing: 2em; }
            50% { opacity: 1; letter-spacing: 0.5em; }
            100% { opacity: 0; letter-spacing: 0.1em; }
        }
    `;
    document.head.appendChild(style);
});