// 取得 DOM 元素
const chatCard = document.getElementById('chat-card');

// ==============================
// 1. 登入成功後的邏輯 (請將這段放入你原本的登入成功 callback 中)
// ==============================
function onLoginSuccess() {
    // 顯示右下角圖示
    launcherBtn.style.display = 'flex'; 
    
    // (選項) 登入後是否自動彈出聊天視窗？
    // chatCard.style.display = 'flex'; // 這是為了讓元素存在
    // setTimeout(() => chatCard.classList.add('open'), 100); 
}

// ==============================
// 2. 登出邏輯 (請放入你原本的登出 callback 中)
// ==============================
function onLogout() {
    chatCard.classList.remove('open'); // 關閉視窗
    setTimeout(() => {
        chatCard.style.display = 'none'; // 隱藏視窗 DOM
        launcherBtn.style.display = 'none'; // 隱藏圖示
    }, 300);
}

// ==============================
// 3. 按鈕互動事件綁定
// ==============================

// 點擊「貓咪圖示」 -> 打開聊天室
launcherBtn.addEventListener('click', () => {
    // 先確保 display 不是 none，這樣 transition 動畫才看得到
    chatCard.style.display = 'flex';
    
    // 用 setTimeout 讓瀏覽器有時間渲染 display:flex，再加 class 觸發動畫
    setTimeout(() => {
        chatCard.classList.add('open');
    }, 10);
    
    // 隱藏圓形按鈕 (像 Messenger 一樣打開後圖示消失，或是你可以選擇保留)
    launcherBtn.style.opacity = '0'; 
    launcherBtn.style.pointerEvents = 'none';
});

// 點擊「X 關閉按鈕」 -> 縮小回圖示
closeChatBtn.addEventListener('click', () => {
    chatCard.classList.remove('open');
    
    // 恢復圓形按鈕
    launcherBtn.style.opacity = '1';
    launcherBtn.style.pointerEvents = 'auto';

    // 等動畫跑完再隱藏 DOM (非必要，但比較省效能)
    setTimeout(() => {
        chatCard.style.display = 'none';
    }, 300);
});