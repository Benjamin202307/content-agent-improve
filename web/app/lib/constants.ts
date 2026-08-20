// Match uvicorn's local IPv4 listener; some Windows setups resolve localhost to IPv6 first.
// 8917 is retained by an older local worker on this machine. The current
// Content Agent backend runs on 8918 so the UI always reaches the active code.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8918";
