
// Clear JARVIS configuration cache
if (typeof localStorage !== 'undefined') {
    const cached = localStorage.getItem('jarvis_dynamic_config');
    if (cached) {
        try {
            const config = JSON.parse(cached);
            // Check if cache points to wrong port
            if (config.API_BASE_URL && (config.API_BASE_URL.includes(':8001') || config.API_BASE_URL.includes(':8000'))) {
                localStorage.removeItem('jarvis_dynamic_config');
                console.log('[JARVIS] Cleared stale configuration cache pointing to wrong port');
            }
        } catch (e) {
            // Invalid cache, clear it
            localStorage.removeItem('jarvis_dynamic_config');
            console.log('[JARVIS] Cleared invalid configuration cache');
        }
    }
}
