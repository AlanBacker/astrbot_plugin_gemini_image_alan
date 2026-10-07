const bridge = window.AstrBotPluginPage;

const MAX_USED = 10000;
const AUTO_REFRESH_MS = 30000;
const MODE_TEXT = { disable: "关闭", blacklist: "黑名单", whitelist: "白名单" };

const state = {
  data: null,
  clockOffset: 0, // 服务器时间减本地时间（毫秒），相对时间按服务器时钟计算
  kind: "all",
  query: "",
  sort: "recent",
  loading: false,
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch],
  );
}

function serverNow() {
  return (Date.now() + state.clockOffset) / 1000;
}

function formatAbsolute(ts) {
  return ts ? new Date(ts * 1000).toLocaleString("zh-CN", { hour12: false }) : "";
}

function formatRelative(ts) {
  if (!ts) return "—";
  const diff = ts - serverNow();
  const abs = Math.abs(diff);
  if (abs < 60) return diff > 0 ? "1分钟内" : "刚刚";
  let text;
  if (abs < 3600) text = `${Math.floor(abs / 60)}分钟`;
  else if (abs < 86400) text = `${Math.floor(abs / 3600)}小时`;
  else text = `${Math.floor(abs / 86400)}天`;
  return diff > 0 ? `${text}后` : `${text}前`;
}

function kindText(session) {
  return session.kind === "group" ? "群聊" : "私聊";
}

function displayName(session) {
  if (session.remark) return session.remark;
  if (session.name) return session.name;
  return `${session.kind === "group" ? "群" : "用户"} ${session.target_id}`;
}

function findSession(subject) {
  return state.data?.sessions.find((item) => item.subject === subject);
}

// ---------- 提示 ----------

let toastTimer = 0;
function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.className = isError ? "toast error" : "toast";
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), isError ? 5000 : 2500);
}

// ---------- 对话框（iframe 沙箱里不能用 confirm/prompt） ----------

let dialogHandler = null;

function openDialog({ title, body, confirmText = "确定", danger = false, onConfirm }) {
  $("dialog-title").textContent = title;
  $("dialog-body").innerHTML = body;
  const confirm = $("dialog-confirm");
  confirm.textContent = confirmText;
  confirm.className = `btn ${danger ? "btn-danger" : "btn-primary"}`;
  confirm.disabled = false;
  $("dialog-cancel").disabled = false;
  $("dialog-error").hidden = true;
  $("dialog").hidden = false;
  dialogHandler = onConfirm;
  const input = $("dialog-body").querySelector("input");
  if (input) {
    input.focus();
    input.select();
  } else {
    confirm.focus();
  }
}

function closeDialog() {
  $("dialog").hidden = true;
  dialogHandler = null;
}

function dialogError(message) {
  const el = $("dialog-error");
  el.textContent = message;
  el.hidden = false;
}

async function submitDialog() {
  if (!dialogHandler || $("dialog-confirm").disabled) return;
  $("dialog-confirm").disabled = true;
  $("dialog-cancel").disabled = true;
  try {
    const message = await dialogHandler();
    closeDialog();
    if (message) toast(message);
    await load();
  } catch (error) {
    dialogError(error?.message || String(error));
    $("dialog-confirm").disabled = false;
    $("dialog-cancel").disabled = false;
  }
}

$("dialog-cancel").addEventListener("click", closeDialog);
$("dialog-confirm").addEventListener("click", submitDialog);
$("dialog").addEventListener("click", (event) => {
  if (event.target === $("dialog")) closeDialog();
});
$("dialog-body").addEventListener("click", (event) => {
  const input = $("used-input");
  const button = event.target.closest("button");
  if (!input || !button) return;
  const current = Number.parseInt(input.value, 10) || 0;
  if (button.dataset.step) {
    input.value = Math.min(MAX_USED, Math.max(0, current + Number(button.dataset.step)));
  } else if (button.dataset.set) {
    input.value = button.dataset.set;
  }
});
document.addEventListener("keydown", (event) => {
  if ($("dialog").hidden) return;
  if (event.key === "Escape") closeDialog();
  if (event.key === "Enter" && event.target.tagName === "INPUT") {
    event.preventDefault();
    submitDialog();
  }
});

// ---------- 渲染 ----------

function renderStats() {
  const { sessions, limits, permission_mode: mode } = state.data;
  const groups = sessions.filter((item) => item.kind === "group").length;
  const used = sessions.reduce((sum, item) => sum + item.usage.day, 0);
  const pending = sessions.reduce((sum, item) => sum + item.usage.pending, 0);
  const listed = sessions.filter((item) => item.listed).length;

  let modeSub = "所有会话均可使用";
  if (mode === "blacklist") modeSub = `已拉黑 ${listed} 个会话`;
  if (mode === "whitelist") modeSub = `白名单内 ${listed} 个会话`;

  const cards = [
    ["会话", sessions.length, `群聊 ${groups} · 私聊 ${sessions.length - groups}`],
    ["近24小时成功生图", used, pending ? `正在执行 ${pending}` : "仅统计成功发送的图片"],
    [
      "频率限制",
      limits.enabled ? `${limits.day} 次 / 24小时` : "未启用",
      `每分钟 ${limits.minute} · 每小时 ${limits.hour}`,
    ],
    ["权限模式", MODE_TEXT[mode] || mode, modeSub],
  ];
  $("stats").innerHTML = cards
    .map(
      ([label, value, sub]) => `
        <div class="stat">
          <div class="stat-label">${esc(label)}</div>
          <div class="stat-value">${esc(value)}</div>
          <div class="stat-sub">${esc(sub)}</div>
        </div>`,
    )
    .join("");
}

function renderNotices(errorMessage = "") {
  const notices = [];
  if (errorMessage) notices.push(`<div class="notice error">${esc(errorMessage)}</div>`);
  if (state.data && !state.data.limits.enabled) {
    notices.push(
      '<div class="notice">频率限制未启用，额度不会拦截请求。这里的修改仍会保存，在插件配置中启用后生效。</div>',
    );
  }
  if (state.data?.warning) notices.push(`<div class="notice">${esc(state.data.warning)}</div>`);
  $("notices").innerHTML = notices.join("");
}

function visibleSessions() {
  const query = state.query.trim().toLowerCase();
  const sortKey = {
    recent: (item) => item.last_seen || 0,
    day: (item) => item.usage.day + item.usage.pending,
    total: (item) => item.total_success || 0,
  }[state.sort];
  return state.data.sessions
    .filter((item) => state.kind === "all" || item.kind === state.kind)
    .filter(
      (item) =>
        !query ||
        [item.remark, item.name, item.target_id, item.last_user_name, item.last_user_id, item.platform]
          .some((value) => String(value || "").toLowerCase().includes(query)),
    )
    .sort((a, b) => sortKey(b) - sortKey(a) || (b.last_seen || 0) - (a.last_seen || 0));
}

function permissionCell(session, mode) {
  let chip = '<span class="chip">允许</span>';
  let sub = "";
  if (mode === "blacklist" && session.listed) {
    chip = '<span class="chip denied">已拉黑</span>';
  } else if (mode === "whitelist") {
    chip = session.listed
      ? '<span class="chip">白名单</span>'
      : '<span class="chip denied">未授权</span>';
    if (!session.listed) sub = "不在白名单";
  } else if (mode === "disable") {
    sub = "未启用名单";
  }
  return `${chip}${sub ? `<div class="perm-sub">${sub}</div>` : ""}`;
}

function permissionButton(session, mode) {
  if (mode === "blacklist") {
    return session.listed
      ? `<button type="button" class="btn btn-sm" data-action="permission">解除拉黑</button>`
      : `<button type="button" class="btn btn-sm btn-danger-outline" data-action="permission">拉黑</button>`;
  }
  if (mode === "whitelist") {
    return session.listed
      ? `<button type="button" class="btn btn-sm btn-danger-outline" data-action="permission">移出白名单</button>`
      : `<button type="button" class="btn btn-sm" data-action="permission">加入白名单</button>`;
  }
  return "";
}

function windowCount(used, pending, limit) {
  const full = used + pending >= limit;
  return `<span class="count${full ? " full" : ""}">${used}<span class="limit"> / ${limit}</span></span>`;
}

function renderRow(session) {
  const { limits, permission_mode: mode } = state.data;
  const usage = session.usage;
  const percent = Math.min(100, (usage.day / limits.day) * 100);
  const barClass = usage.day + usage.pending >= limits.day ? "full" : percent >= 80 ? "warn" : "";

  const meta = [`ID ${session.target_id}`];
  if (session.platform) meta.push(session.platform);
  if (session.remark && session.name) meta.push(session.name);
  if (session.kind === "group" && session.last_user_name) {
    meta.push(`最近使用者 ${session.last_user_name}`);
  }

  const usageSub = [];
  if (usage.pending) usageSub.push(`<span class="pending">正在执行 ${usage.pending}</span>`);
  usageSub.push(
    usage.next_release
      ? `<span title="${esc(formatAbsolute(usage.next_release))}">${formatRelative(usage.next_release)}恢复 1 次</span>`
      : "额度未使用",
  );

  return `
    <div class="row" data-subject="${esc(session.subject)}">
      <div class="session">
        <span class="kind ${session.kind}" title="${kindText(session)}">${session.kind === "group" ? "群" : "私"}</span>
        <div class="session-text">
          <div class="session-name" title="${esc(displayName(session))}">${esc(displayName(session))}</div>
          <div class="session-meta" title="${esc(meta.join(" · "))}">${esc(meta.join(" · "))}</div>
        </div>
      </div>
      <div><span class="cell-label">权限</span>${permissionCell(session, mode)}</div>
      <div><span class="cell-label">最近1分钟</span>${windowCount(usage.minute, usage.pending, limits.minute)}</div>
      <div><span class="cell-label">最近1小时</span>${windowCount(usage.hour, usage.pending, limits.hour)}</div>
      <div class="day-cell">
        <span class="cell-label">滚动24小时</span>
        ${windowCount(usage.day, usage.pending, limits.day)}
        <div class="bar"><span class="${barClass}" style="width: ${percent}%"></span></div>
        <div class="usage-sub">${usageSub.join(" · ")}</div>
      </div>
      <div><span class="cell-label">累计成功</span><span class="count">${session.total_success || 0}</span></div>
      <div title="${esc(formatAbsolute(session.last_seen))}">
        <span class="cell-label">最近使用</span>${formatRelative(session.last_seen)}
      </div>
      <div class="actions">
        <button type="button" class="btn btn-sm" data-action="usage">修改次数</button>
        <button type="button" class="btn btn-sm" data-action="reset"${usage.day ? "" : " disabled"}>重置</button>
        <button type="button" class="btn btn-sm" data-action="remark">备注</button>
        ${permissionButton(session, mode)}
      </div>
    </div>`;
}

function renderList() {
  const list = $("list");
  if (!state.data) return;
  if (!state.data.sessions.length) {
    list.innerHTML =
      '<div class="empty">还没有会话记录。群聊或私聊使用 /生图 或让 AI 调用生图工具后，会话会出现在这里。</div>';
    return;
  }
  const sessions = visibleSessions();
  if (!sessions.length) {
    list.innerHTML = '<div class="empty">没有符合条件的会话</div>';
    return;
  }
  list.innerHTML = `
    <div class="list-head">
      <div>会话</div><div>权限</div><div>最近1分钟</div><div>最近1小时</div>
      <div>滚动24小时</div><div>累计成功</div><div>最近使用</div><div></div>
    </div>
    ${sessions.map(renderRow).join("")}`;
}

function render() {
  renderStats();
  renderNotices();
  renderList();
}

// ---------- 数据 ----------

async function load() {
  if (state.loading) return;
  state.loading = true;
  $("refresh").disabled = true;
  try {
    const data = await bridge.apiGet("sessions");
    state.data = data;
    state.clockOffset = data.now * 1000 - Date.now();
    render();
  } catch (error) {
    renderNotices(`加载失败: ${error?.message || error}`);
    if (!state.data) $("list").innerHTML = '<div class="empty">加载失败</div>';
  } finally {
    state.loading = false;
    $("refresh").disabled = false;
  }
}

// ---------- 操作 ----------

function editUsage(session) {
  const { limits } = state.data;
  const pendingHint = session.usage.pending
    ? `<p class="hint">另有 ${session.usage.pending} 个任务正在执行，成功后仍会计入。</p>`
    : "";
  openDialog({
    title: "修改已用次数",
    body: `
      <p>${esc(displayName(session))} 在滚动24小时内已成功生图 <b>${session.usage.day}</b> 次，上限 ${limits.day} 次。</p>
      <div class="stepper">
        <button type="button" class="btn" data-step="-1" aria-label="减一">−</button>
        <input type="number" class="input" id="used-input" min="0" max="${MAX_USED}" step="1"
          value="${session.usage.day}" aria-label="已用次数" />
        <button type="button" class="btn" data-step="1" aria-label="加一">+</button>
      </div>
      <div class="quick">
        <button type="button" class="btn btn-sm" data-set="0">清零</button>
        <button type="button" class="btn btn-sm" data-set="${limits.day}">用满（${limits.day}）</button>
      </div>
      <p class="hint">只调整滚动24小时的计数，尽量不影响每分钟、每小时的限制。减少时先撤销最早的记录；补记的次数约23小时后自动释放。</p>
      ${pendingHint}`,
    confirmText: "保存",
    onConfirm: async () => {
      const raw = $("used-input").value.trim();
      const used = Number(raw);
      if (!/^\d+$/.test(raw) || used > MAX_USED) {
        throw new Error(`请输入 0 到 ${MAX_USED} 之间的整数`);
      }
      await bridge.apiPost("sessions/usage", { subject: session.subject, used });
      return `已将 ${displayName(session)} 的已用次数改为 ${used}`;
    },
  });
}

function resetUsage(session) {
  openDialog({
    title: "重置额度",
    body: `<p>清除 ${esc(displayName(session))} 在滚动24小时内的 ${session.usage.day} 次成功记录，立即恢复全部额度。</p>
      <p class="hint">正在执行的任务不受影响，累计成功次数保留。</p>`,
    confirmText: "重置",
    danger: true,
    onConfirm: async () => {
      await bridge.apiPost("sessions/reset", { subject: session.subject });
      return `已重置 ${displayName(session)} 的额度`;
    },
  });
}

function resetAll() {
  if (!state.data) return;
  const used = state.data.sessions.filter((item) => item.usage.day > 0);
  const total = used.reduce((sum, item) => sum + item.usage.day, 0);
  if (!used.length) {
    toast("当前没有需要重置的额度");
    return;
  }
  openDialog({
    title: "重置全部额度",
    body: `<p>清除全部 ${used.length} 个会话在滚动24小时内的成功记录，共 ${total} 次。此操作无法撤销。</p>
      <p class="hint">正在执行的任务不受影响，累计成功次数保留。</p>`,
    confirmText: "全部重置",
    danger: true,
    onConfirm: async () => {
      await bridge.apiPost("sessions/reset_all", {});
      return "已重置全部会话的额度";
    },
  });
}

function editRemark(session) {
  openDialog({
    title: "会话备注",
    body: `
      <p>${kindText(session)} ${esc(session.target_id)}${session.name ? `（${esc(session.name)}）` : ""}</p>
      <input type="text" class="input field" id="remark-input" maxlength="64"
        value="${esc(session.remark)}" placeholder="例如：项目群、测试账号" aria-label="备注" />
      <p class="hint">备注只在本页面显示，方便辨认会话。留空则清除。</p>`,
    confirmText: "保存",
    onConfirm: async () => {
      const remark = $("remark-input").value.trim();
      await bridge.apiPost("sessions/remark", { subject: session.subject, remark });
      return remark ? "备注已保存" : "备注已清除";
    },
  });
}

async function togglePermission(session) {
  const mode = state.data.permission_mode;
  const listed = !session.listed;
  const listName = mode === "blacklist" ? "黑名单" : "白名单";
  const action = async () => {
    await bridge.apiPost("sessions/permission", { subject: session.subject, listed });
    return `已${listed ? "加入" : "移出"}${listName}`;
  };
  // 拉黑、移出白名单会让会话无法使用，先确认；放行操作直接执行
  const blocks = mode === "blacklist" ? listed : !listed;
  if (blocks) {
    openDialog({
      title: mode === "blacklist" ? "拉黑会话" : "移出白名单",
      body: `<p>操作后，${kindText(session)} ${esc(displayName(session))} 将无法使用生图功能。</p>
        <p class="hint">会同步修改插件配置中的${session.kind === "group" ? "群组" : "用户"}名单。</p>`,
      confirmText: mode === "blacklist" ? "拉黑" : "移出",
      danger: true,
      onConfirm: action,
    });
    return;
  }
  try {
    toast(await action());
    await load();
  } catch (error) {
    toast(error?.message || String(error), true);
  }
}

$("list").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  const row = event.target.closest(".row");
  if (!button || !row) return;
  const session = findSession(row.dataset.subject);
  if (!session) return;
  const handlers = {
    usage: editUsage,
    reset: resetUsage,
    remark: editRemark,
    permission: togglePermission,
  };
  handlers[button.dataset.action]?.(session);
});

$("refresh").addEventListener("click", () => load());
$("reset-all").addEventListener("click", resetAll);
$("search").addEventListener("input", (event) => {
  state.query = event.target.value;
  if (state.data) renderList();
});
$("sort").addEventListener("change", (event) => {
  state.sort = event.target.value;
  if (state.data) renderList();
});
$("kind-filter").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-kind]");
  if (!button) return;
  state.kind = button.dataset.kind;
  for (const item of $("kind-filter").querySelectorAll("button")) {
    item.setAttribute("aria-checked", String(item === button));
  }
  if (state.data) renderList();
});

// ---------- 启动 ----------

async function main() {
  if (!bridge) {
    $("list").innerHTML =
      '<div class="empty">请在 AstrBot WebUI 的插件详情页中打开此页面（需要 AstrBot v4.24.1 或更高版本）。</div>';
    return;
  }
  await bridge.ready();
  await load();
  setInterval(() => {
    if (document.visibilityState === "visible" && $("dialog").hidden) load();
  }, AUTO_REFRESH_MS);
}

main();
