/**
 * Clears all application state to prevent data leakage between demo roles.
 */
export const clearAppState = () => {
  const demoFlag = localStorage.getItem('demo_session');
  const theme = localStorage.getItem('theme');
  
  localStorage.clear();
  sessionStorage.clear();
  
  // Restore necessary flags
  if (demoFlag) localStorage.setItem('demo_session', demoFlag);
  if (theme) localStorage.setItem('theme', theme);
};

export const handleDemoTransition = async (role, loginFn, authApiFn) => {
  try {
    clearAppState();
    const res = await authApiFn(role);
    if (res.data && res.data.access_token) {
      // The loginFn will set state, but we'll force a reload anyway to be safe
      // but we need to set the storage first so the reload catches it
      const { user, access_token } = res.data;
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('user', JSON.stringify(user));
      localStorage.setItem('demo_session', 'true');
      
      return true;
    }
  } catch (err) {
    console.error('Demo transition failed:', err);
    throw err;
  }
  return false;
};
