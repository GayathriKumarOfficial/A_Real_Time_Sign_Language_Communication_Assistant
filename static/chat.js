/**
 * chat.js
 * Shared chat UI utilities used by both deaf_dashboard.html and hearing_dashboard.html
 * Handles: message rendering, bubble styling, auto-scroll, system messages
 */

/**
 * Append a chat message bubble to the chat window.
 * @param {Object} msg     - { text, sender, role, timestamp, sid }
 * @param {string} myRole  - 'deaf' | 'hearing'  (the current user's role)
 */
function appendMessage(msg, myRole) {
  const chatWindow = document.getElementById('chatWindow');
  const chatEmpty  = document.getElementById('chatEmpty');

  // Hide empty state
  if (chatEmpty) chatEmpty.style.display = 'none';

  // Determine outgoing vs incoming
  const isOutgoing = (msg.role === myRole);
  const direction  = isOutgoing ? 'outgoing' : 'incoming';

  // Role-specific colour class for outgoing bubbles
  let roleClass = '';
  if (isOutgoing) {
    roleClass = msg.role === 'deaf' ? 'deaf-outgoing' : 'hearing-outgoing';
  }

  // Build bubble
  const msgEl = document.createElement('div');
  msgEl.className = `chat-msg ${direction} ${roleClass}`;

  // Sender label + timestamp (show above for incoming, below for outgoing)
  const metaEl = document.createElement('div');
  metaEl.className = 'msg-meta';

  const senderEl = document.createElement('span');
  senderEl.className = 'msg-sender';
  senderEl.textContent = isOutgoing ? 'You' : (msg.sender || 'Other');

  const timeEl = document.createElement('span');
  timeEl.className = 'msg-time';
  timeEl.textContent = msg.timestamp || '';

  metaEl.appendChild(senderEl);
  metaEl.appendChild(timeEl);

  // Bubble text
  const bubbleEl = document.createElement('div');
  bubbleEl.className = 'bubble';
  bubbleEl.textContent = msg.text;

  // Stack: meta on top for incoming, bubble then meta for outgoing
  if (isOutgoing) {
    msgEl.appendChild(bubbleEl);
    msgEl.appendChild(metaEl);
  } else {
    msgEl.appendChild(metaEl);
    msgEl.appendChild(bubbleEl);
  }

  chatWindow.appendChild(msgEl);
  scrollToBottom(chatWindow);
}

/**
 * Append a system / info message (centered pill).
 * @param {string} text
 */
function appendSystemMessage(text) {
  const chatWindow = document.getElementById('chatWindow');
  if (!chatWindow) return;

  const el = document.createElement('div');
  el.className = 'system-msg';
  el.textContent = text;

  chatWindow.appendChild(el);
  scrollToBottom(chatWindow);
}

/**
 * Smooth-scroll a container to its bottom.
 * @param {HTMLElement} el
 */
function scrollToBottom(el) {
  el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
}