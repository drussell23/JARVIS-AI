// JARVIS Landing Page - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Navbar background on scroll
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(10, 10, 10, 0.95)';
            navbar.style.backdropFilter = 'blur(20px)';
        } else {
            navbar.style.background = 'rgba(10, 10, 10, 0.9)';
            navbar.style.backdropFilter = 'blur(10px)';
        }
    });

    // Reveal animations on scroll
    const revealElements = document.querySelectorAll('.feature-card, .capability-item, .step-card');
    
    const revealOnScroll = function() {
        const windowHeight = window.innerHeight;
        revealElements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < windowHeight - elementVisible) {
                element.classList.add('reveal', 'active');
            }
        });
    };

    window.addEventListener('scroll', revealOnScroll);
    revealOnScroll(); // Check on load

    // Copy code functionality
    window.copyCode = function(button) {
        const codeBlock = button.previousElementSibling;
        const code = codeBlock.textContent;
        
        navigator.clipboard.writeText(code).then(() => {
            button.classList.add('copied');
            const originalHTML = button.innerHTML;
            button.innerHTML = '<i class="fas fa-check"></i>';
            
            setTimeout(() => {
                button.classList.remove('copied');
                button.innerHTML = originalHTML;
            }, 2000);
        });
    };

    // Typing effect for command preview
    const commandPreview = document.querySelector('.command-preview');
    if (commandPreview) {
        const commands = [
            '> "Hey JARVIS, open Chrome"',
            '> "Set volume to 50%"',
            '> "Take a screenshot"',
            '> "Start morning routine"',
            '> "Search for Python tutorials"',
            '> "Close all distractions"',
            '> "Show system status"'
        ];
        
        let currentCommand = 0;
        let currentChar = 0;
        let isDeleting = false;
        
        function typeCommand() {
            const command = commands[currentCommand];
            
            if (!isDeleting) {
                commandPreview.textContent = command.substring(0, currentChar);
                currentChar++;
                
                if (currentChar > command.length) {
                    isDeleting = true;
                    setTimeout(typeCommand, 2000);
                    return;
                }
            } else {
                commandPreview.textContent = command.substring(0, currentChar);
                currentChar--;
                
                if (currentChar === 0) {
                    isDeleting = false;
                    currentCommand = (currentCommand + 1) % commands.length;
                }
            }
            
            const typingSpeed = isDeleting ? 50 : 100;
            setTimeout(typeCommand, typingSpeed);
        }
        
        typeCommand();
    }

    // Hologram rotation based on mouse position
    const hologramContainer = document.querySelector('.hologram-container');
    if (hologramContainer) {
        document.addEventListener('mousemove', function(e) {
            const x = (e.clientX / window.innerWidth - 0.5) * 20;
            const y = (e.clientY / window.innerHeight - 0.5) * 20;
            
            hologramContainer.style.transform = `rotateY(${x}deg) rotateX(${-y}deg)`;
        });
    }

    // Wave animation for voice visualizer
    const waveBars = document.querySelectorAll('.wave-bar');
    if (waveBars.length > 0) {
        setInterval(() => {
            waveBars.forEach(bar => {
                const height = Math.random() * 40 + 20;
                bar.style.height = `${height}px`;
            });
        }, 300);
    }

    // Tech card floating animation enhancement
    const techCards = document.querySelectorAll('.tech-card');
    techCards.forEach((card, index) => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-30px) scale(1.05)';
            this.style.boxShadow = '0 20px 40px rgba(0, 212, 255, 0.4)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            this.style.boxShadow = 'none';
        });
    });

    // Arc reactor pulse synchronization
    const arcReactors = document.querySelectorAll('.arc-reactor-large, .arc-reactor-mini');
    arcReactors.forEach(reactor => {
        reactor.addEventListener('mouseenter', function() {
            this.style.animationDuration = '1s';
        });
        
        reactor.addEventListener('mouseleave', function() {
            this.style.animationDuration = '';
        });
    });

    // Capability item hover effects
    const capabilityItems = document.querySelectorAll('.capability-item');
    capabilityItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            const category = this.getAttribute('data-category');
            // Could trigger different animations based on category
            this.style.borderColor = 'var(--arc-blue)';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.borderColor = 'transparent';
        });
    });

    // Loading animation removal
    window.addEventListener('load', function() {
        document.body.classList.add('loaded');
        
        // Start animations after load
        setTimeout(() => {
            document.querySelectorAll('.glitch').forEach(element => {
                element.classList.add('active');
            });
        }, 500);
    });

    // Mobile menu toggle (if needed)
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', function() {
            document.querySelector('.nav-links').classList.toggle('active');
        });
    }

    // Intersection Observer for advanced animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all animated elements
    document.querySelectorAll('.feature-card, .capability-item, .step-card').forEach(el => {
        observer.observe(el);
    });

    // Easter egg: Konami code for special effect
    const konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
    let konamiIndex = 0;

    document.addEventListener('keydown', function(e) {
        if (e.key === konamiCode[konamiIndex]) {
            konamiIndex++;
            if (konamiIndex === konamiCode.length) {
                activateJarvisMode();
                konamiIndex = 0;
            }
        } else {
            konamiIndex = 0;
        }
    });

    function activateJarvisMode() {
        document.body.classList.add('jarvis-mode');
        const message = document.createElement('div');
        message.className = 'jarvis-message';
        message.textContent = 'JARVIS Protocol Activated';
        document.body.appendChild(message);
        
        setTimeout(() => {
            message.remove();
            document.body.classList.remove('jarvis-mode');
        }, 3000);
    }
});