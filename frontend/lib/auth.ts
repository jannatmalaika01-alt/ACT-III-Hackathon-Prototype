// DEMO ONLY: fake login. Do not call this "authentication" in the pitch.
const KEY = "act3_user";

export function login(username: string, password: string): boolean {
  if (username === "admin" && password === "factory123") {
    localStorage.setItem(KEY, username);
    return true;
  }
  return false;
}
export const getUser = () => (typeof window === "undefined" ? null : localStorage.getItem(KEY));
export const logout = () => localStorage.removeItem(KEY);