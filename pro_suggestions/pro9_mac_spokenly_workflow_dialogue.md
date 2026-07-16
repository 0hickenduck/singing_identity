我有个关于 MacBook 的问题，还有一个关于 spoken English 的问题。关于 MacBook 的问题是，我现在在用 MacBook Air 嘛，最新版的。然后我只有一个屏幕，然后我会经常在不同的软件里面切换。具体来说，会在浏览器的不同的 tab 里面切换，会在 不同浏览器之间切换。然后有时候会不停的看新的，有没有新的 email。想看新喵的内容，看 Slack 看 Obsidian 看 ToDoist 看 Terminal 看 VS Code 看 AntiGravity 看 Codex。如果我有多屏的话，这个事情没有那么麻烦。或者说多屏加上那个虚拟桌面，就 Windows 虚拟桌面。我以前是这么工作的，虽然也没有很顺畅。现在的话在 Mac 上面，如果我的屏幕比较少，我可以用那个四指滑动。就是每个单独的窗口开一个桌面，然后四指滑动。但我的桌，窗口太多了，而且四指滑动没办法在那个 Tab 之间滑动。然后我现在基本上都是四指上滑，然后去选。也比较慢。就 friction 比较大，应该这么说。尤其是当我要两个东西对照着看的时候。所以我想问有没有什么。想法，方法帮我改进。第二个问题是关于 Spokenly 的。Spokenly 呢，这里有个 Dication models 的 setting。我把它 setting 成那个 另外一个 model 之后，它仍然会显示我，就是或，它仍然用的是 eleven five。但是我如果点到它里面的 history 然后点 Retry 的话，它用的就是我选的那个 model 啊。所以我很好奇，这个到底是怎么回事？可能什么原因？就是 我选的是另外一个模型，但是我用， Dictation 的时候，实际上用的是， ByteDance Lab 但是我如果点到 history 里面，用 Retry 的话，就是我选的那个模型。我怎么样在， Dictation 的时候，就用我选的那个模型？然后以及，MacBook 就只是我说的那几个事情。你可以发现我是很喜欢多开，然后在不同画面里面切换，同时做很多事情。就，你有没有类似的，操作我需要学会的？或者说有没有什么想法我来？更好的去工作。比如说如果我只在 terminal 里面工作的话，我可以用 tmux 那 tmux 的话可以分屏，可以换。或者说你觉得我这个分屏的思路本身就有问题。但有时候确实我需要从另外一个软件里面得到信息，比如说我一边开 AI 一边开 LeeTcode 的 coding。等等等等 就当我需要这样做的时候，我应该怎么做？以及 你有什么别的想法就是我应该怎么做？比如说你可以建议我更少的这么做之类的，但我觉得这么做本身没有什么问题。我有时候就是需要很多的信息同步，比如说我又要做，看论文，又要一边有一个 AI 带着我去读这个论文，又要一边去做，不是一点，那我屏幕不够这么大。

---
你现在的卡点不是“多开”本身，而是**你在用视觉搜索来找上下文**：四指上滑、看一堆窗口、再点进去。这在窗口超过 8–10 个之后一定会变慢。你需要把 Mac 调成更像 tmux：**固定编号、固定布局、固定入口**，少用“看一眼再找”。

## 1. MacBook 单屏：不要“一个窗口一个桌面”，要“一个任务一个桌面”

我建议你把 Spaces 控制在 **4–5 个**，每个 Space 是一个工作上下文，而不是一个 app。

一个适合你的布局可以是：

|Space|角色|放什么|
|---|---|---|
|`Ctrl+1`|**Build / Code**|VS Code、Terminal、Codex、Antigravity|
|`Ctrl+2`|**Reference / AI Pair**|浏览器/PDF/LeetCode + AI，对照阅读或做题|
|`Ctrl+3`|**Research / Browser**|论文、搜索、多个 browser windows|
|`Ctrl+4`|**Comms / Inbox**|Mail、Slack、ToDoist|
|`Ctrl+5`|**Notes / Memory**|Obsidian、草稿、长文整理|

macOS 的 Spaces 本来就是用来把窗口按桌面组织的；Apple 文档里也说明了可以创建多个 space、在 space 间切换、把 app 指定到某个 space。切换相邻 space 可以用 `Control + ←/→`，App 也可以通过 Dock 的 Options 分配到某个桌面。([Apple Support](https://support.apple.com/guide/mac-help/work-in-multiple-spaces-mh14112/mac "Work in multiple spaces on Mac - Apple Support"))

最重要的设置是：**启用 `Ctrl+1`、`Ctrl+2`、`Ctrl+3` 这种直接跳桌面快捷键**。这比四指滑动快很多，因为它不是“翻页”，而是“寻址”。通常路径是：

`System Settings → Keyboard → Keyboard Shortcuts → Mission Control → 勾选 Switch to Desktop 1/2/3...`

注意：对应的 Desktop 必须先创建出来，设置里才会出现 `Switch to Desktop N`。这个快捷键默认经常没有启用。([SpaceJump](https://www.getspacejump.com/guides/mission-control-shortcuts?utm_source=chatgpt.com "Every Mission Control keyboard shortcut on Mac (2026)"))

同时建议关掉：

`System Settings → Desktop & Dock → Mission Control → Automatically rearrange Spaces based on most recent use`

否则你的 `Ctrl+2` 今天是论文，明天可能变成 Slack，这会毁掉肌肉记忆。

## 2. 单屏对照阅读：用“二窗布局”，不要全屏堆叠

你说的“AI + LeetCode”“AI + 论文”“AI + Obsidian”这种场景，核心不是多桌面，而是**当前 Space 里稳定两栏**：

- 左边：主任务，比如 LeetCode、PDF、paper、网页。
    
- 右边：辅助任务，比如 ChatGPT、Claude、Codex、Obsidian note。
    
- 需要 terminal 时，不要再开一个新桌面；用 VS Code terminal / tmux / 一个窄 terminal pane。
    

新版 macOS 已经有原生窗口平铺快捷键，例如 `Fn + Control + ←/→` 把当前窗口放到左右半屏，`Fn + Control + Shift + ←/→` 可以把两个窗口排成左右组合。Apple 的窗口平铺文档列出了这些快捷键。([Apple Support](https://support.apple.com/guide/mac-help/mchl9674d0b0/mac "Mac window tiling icons & keyboard shortcuts - Apple Support (NZ)"))

但我会更推荐你装一个窗口管理器，因为你的需求明显超过原生平铺：

- **Rectangle**：免费开源，用快捷键或拖到边缘来移动/缩放窗口，适合左右半屏、三分之一、角落等布局。([Rectangle](https://rectangleapp.com/ "Rectangle"))
    
- **Raycast Window Management**：如果你已经用 Raycast，它的窗口管理扩展可以设置全局热键，把窗口移动到左半、右半、三分之一、四分之一、居中、最大化等。([Raycast](https://www.raycast.com/core-features/window-management "Mac Window Manager Extension | Raycast"))
    

我的建议是：**不要依赖鼠标拖窗口**。把这些变成固定快捷键，例如：

|快捷键|动作|
|---|---|
|`Hyper + H`|当前窗口左半屏|
|`Hyper + L`|当前窗口右半屏|
|`Hyper + K`|当前窗口上半屏|
|`Hyper + J`|当前窗口下半屏|
|`Hyper + F`|最大化但不全屏|
|`Hyper + C`|居中|

这里的 `Hyper` 可以是 `Caps Lock = Control + Option + Command + Shift`。Karabiner-Elements 是 macOS 上常用的键盘自定义工具，支持这种重映射。([Karabiner-Elements](https://karabiner-elements.pqrs.org/?utm_source=chatgpt.com "Karabiner-Elements"))

## 3. 你最该学的 Mac 切换快捷键

你现在四指上滑太多了。先把这几个变成肌肉记忆：

|快捷键|用途|
|---|---|
|`Command + Tab`|切换 app|
|`Command + ``|切换当前 app 的不同窗口|
|`Control + Up`|Mission Control|
|`Control + Down`|当前 app 的所有窗口|
|`Control + ←/→`|左右切 Spaces|
|`Control + 1/2/3/4/5`|直接跳到固定 Space|
|浏览器里 `Command + 1..9`|直接跳 tab|
|浏览器里 `Control + Tab` / `Shift + Control + Tab`|前后 tab，具体看 app 菜单|

Apple 官方快捷键文档确认了 `Command + Tab` 是在打开的 app 中切换，`Command + \`` 是切换当前 app 的窗口，`Control + Up `打开 Mission Control，`Control + Down` 显示当前 app 的所有窗口。([Apple Support](https://support.apple.com/en-us/102650 "Mac keyboard shortcuts - Apple Support"))

但是 macOS 原生 `Command + Tab` 的一个问题是：它切的是 **app**，不是窗口。你以前 Windows 的 `Alt+Tab` 更像是“窗口级切换”。所以你很可能会喜欢 **AltTab for macOS**，它就是把 Windows 风格的窗口切换带到 macOS。([GitHub](https://github.com/lwouis/alt-tab-macos "GitHub - lwouis/alt-tab-macos: Windows alt-tab on macOS · GitHub"))

我会这样配：

- `Command + Tab`：只在 app 间粗切。
    
- `Option + Tab` 或 `Hyper + Tab`：AltTab，窗口级切换。
    
- `Control + 1..5`：任务上下文切换。
    
- `Command + ``：同一个 app 内切窗口，比如多个 VS Code / Chrome windows。
    

这会比 Mission Control 视觉挑选快很多。

## 4. 浏览器 tabs：不要让 tab 成为你的第二个桌面系统

你现在有“浏览器不同 tab、不同浏览器、不同软件、不同 Spaces”四层切换系统。摩擦大，是因为层级太多。

我建议你这样分：

**不要按浏览器分，而是按项目分。**

比如：

- 一个 browser window 叫 `paper-reading`
    
- 一个 browser window 叫 `leetcode`
    
- 一个 browser window 叫 `general-search`
    
- 一个 browser window 叫 `admin / email / docs`
    

每个 window 里面再有 tabs。这样你切换的是“项目窗口”，不是在 80 个 tabs 里找。

如果你确实需要多个浏览器，比如 Chrome / Safari / Arc / Firefox，最好让它们有明确职责：

|浏览器|角色|
|---|---|
|主浏览器|日常搜索、AI、文档|
|第二浏览器|登录隔离、测试、另一个账号|
|第三浏览器|临时、不保存状态|

否则“不同浏览器之间切换”会变成无意义的 context switching。

## 5. Slack / Email / ToDoist：不要散落在每个 Space

你说你会不停看有没有新 email。这个行为的 friction 很高，而且会污染所有工作空间。

我建议把 Mail、Slack、ToDoist 全部放到 `Ctrl+4` 的 Comms space。你要看消息时：

1. `Ctrl+4`
    
2. 看完
    
3. `Ctrl+1` 或 `Ctrl+2` 回去
    

这样你的大脑会知道：**通信只发生在 4 号桌面**。

不要把 Slack 窗口开在 coding space，不要把 email tab 混在 paper-reading browser 里。这样你会不断被上下文拖走。

紧急消息靠通知，不靠你手动轮询。非紧急消息用批处理。

## 6. Stage Manager 可以试，但我不建议作为主系统

Stage Manager 的设计是把当前 app 放中间，最近使用 app 放左边；它也支持把多个 app 分成组。Apple 文档里说明了 Stage Manager 可以把相关 app 组合起来，并在组之间切换。([Apple Support](https://support.apple.com/guide/mac-help/use-stage-manager-mchl534ba392/mac "Organize your Mac desktop with Stage Manager - Apple Support"))

但对你这种“代码 + terminal + AI + 浏览器 + 论文 + notes”的人，Stage Manager 很容易变成另一个视觉选择器。它适合轻量办公，不适合作为你的主工作流。

我会把 Stage Manager 当作实验功能，而不是核心。核心应该是：

**固定 Spaces + 窗口管理器 + AltTab + Hyper key。**

## 7. 更高级：把 macOS 做成 tmux

你已经提到 tmux，这个类比非常好。你可以这样理解：

|tmux|macOS|
|---|---|
|session|Space|
|window|App / project window|
|pane|tiled window|
|prefix key|Hyper key|
|`prefix + 1/2/3`|`Ctrl + 1/2/3`|
|layout preset|Rectangle / Raycast / Hammerspoon|

如果你愿意折腾，**Hammerspoon** 很适合你。它是 macOS 自动化工具，可以用 Lua 控制 app、window、screen、clipboard、keyboard event 等。([hammerspoon.org](https://www.hammerspoon.org/ "Hammerspoon"))

你最后可以做到：

- `Hyper + R`：打开 Research layout，左边 paper，右边 AI，下面 Obsidian。
    
- `Hyper + C`：打开 Coding layout，VS Code 最大化，terminal 右侧，browser 在 Space 2。
    
- `Hyper + M`：跳到 Mail/Slack/ToDoist。
    
- `Hyper + A`：打开当前 AI 工具。
    
- `Hyper + T`：打开 terminal / tmux session。
    

这比“我看到哪个窗口就点哪个”稳定得多。

---

# Spokenly 的问题

你的现象是：

> Settings 里选了另一个 dictation model，但正常 Dictation 仍显示/使用旧模型；History 里 Retry 却会用你选的新模型。

我不能 100% 断言，因为我没有你的 Spokenly 日志和界面截图，但这很像 **实时 dictation 路径** 和 **history retry / file transcription 路径** 使用了不同配置，或者正常 dictation 入口被某个 prompt / quick command / 缓存模型绑住了。

Spokenly 本身支持在线模型和本地模型，也会把 transcription 存到 history；App Store 描述也提到可选 online/offline models、实时转写、history 保存。([App Store](https://apps.apple.com/us/app/spokenly-audio-to-text-ai-app/id6740315592 "‎Spokenly: Audio to Text AI app App - App Store")) 另外，Spokenly 的工作流不只是“语音转文字”一步；常见设置里会分成 speech-to-text 模型和后处理/formatting 的 text model。一个近期使用文也明确说 Spokenly 允许分别指定这两个步骤的模型：第一步是语音识别，第二步是清理和格式化文本。([Zenn](https://zenn.dev/rewse/articles/spokenly-local-ai-voice-input?locale=en "Build a Fully Local AI-Assisted Voice Input System with Spokenly"))

更关键的是，有个第三方 Raycast Spokenly extension 反向研究了 Spokenly 的偏好设置。它提到 Spokenly 偏好里至少有这些 key：`transcriptionModelID`、`fileTranscriptionVoiceModelID`、`mainPrompt`、`recentDictationModels`、`quickCommands`；其中 `transcriptionModelID` 是 active dictation transcription model，`fileTranscriptionVoiceModelID` 是 file transcription model。它还提到 `spokenly://start?prompt_id=<UUID>` 会用某个 prompt 启动 dictation，Quick Commands 也走类似机制。([GitHub](https://github.com/mattiacolombomc/raycast-spokenly "GitHub - mattiacolombomc/raycast-spokenly: Raycast extension for the Spokenly macOS dictation app · GitHub"))

所以我会优先怀疑这几种原因：

## A. 你改的是“可用于 Retry/文件转写”的模型，但实时 Dictation 仍在用另一个实时模型

实时 dictation 对延迟要求很高，不是每个模型都适合 streaming。某些模型可能在 History Retry 时可以跑，因为它是拿保存的 `.wav` 重新转写；但实时 dictation 可能会 fallback 到一个支持实时的模型，例如你看到的 Eleven / ByteDance 那个。

这种情况下，不是你眼花，而是 Spokenly 的 live pipeline 和 retry pipeline 不完全一样。

## B. 你的快捷键启动的不是 Main Dictation，而是某个 Quick Command / Prompt

Raycast extension 的说明里提到 Spokenly 可以通过 `spokenly://start?prompt_id=<UUID>` 启动某个 prompt，Quick Commands 也有自己的 ID。([GitHub](https://github.com/mattiacolombomc/raycast-spokenly "GitHub - mattiacolombomc/raycast-spokenly: Raycast extension for the Spokenly macOS dictation app · GitHub")) 如果你的全局快捷键绑定的是旧 quick command，它可能带着旧模型或旧 prompt 设置。

排查方式：

1. 去 Spokenly 的 Prompt / Quick Commands 设置里看一眼。
    
2. 找到你实际用快捷键触发的那个 command。
    
3. 看它有没有自己的 Advanced Settings / AI Provider / model。
    
4. 暂时不要用快捷键，从 Spokenly 主界面直接开始一次 dictation，看是否使用新模型。
    

如果主界面正确、快捷键错误，那问题就在快捷键绑定的 command。

## C. 你混淆了三个 model

Spokenly 里至少要区分：

|设置|作用|
|---|---|
|Dictation / Speech Recognition Model|把声音转成原始文字|
|AI Provider / Text Model / Prompt model|把原始文字清理、润色、格式化|
|File transcription / History retry model|对已有音频重新转写|

你说 Retry 会用你选的模型，说明这个模型大概率是可用的。问题更像是：**live dictation 没有读到同一个设置**，或者 live dictation 入口另有配置。

## D. Spokenly 仍在用旧偏好缓存

macOS app 有时会缓存 UserDefaults。你可以做一次干净测试：

1. 在 Spokenly 里选目标 model。
    
2. 从 menu bar 里彻底 Quit Spokenly，不只是关窗口。
    
3. 重新打开 Spokenly。
    
4. 不用 History，不用 Retry，直接开始一次普通 dictation。
    
5. 看显示的 model 是不是变了。
    

如果重启后变正常，那就是运行时缓存。

## E. 所选模型没下载好 / API key 不可用 / live mode 不支持，于是 fallback

如果你选的是云模型，确认 API key 已经配置并测试通过。  
如果你选的是本地模型，确认模型已经下载完成。  
如果它只在 Retry 有效、live dictation 不生效，很可能是 live mode 不支持或 fallback 没有提示清楚。

## 8. 你可以用 Terminal 验证 Spokenly 实际保存了什么 model

这个命令可以读 Spokenly 的偏好文件，看看当前保存的模型 ID 是什么：

```bash
for key in transcriptionModelID fileTranscriptionVoiceModelID recentDictationModels; do
  printf "\n%s = " "$key"
  plutil -extract "$key" raw -o - ~/Library/Preferences/app.spokenly.plist 2>/dev/null | base64 -D
  echo
done
```

你主要看：

- `transcriptionModelID`：普通 dictation 应该用的模型。
    
- `fileTranscriptionVoiceModelID`：文件转写/可能与 Retry 相关的模型。
    
- `recentDictationModels`：最近用过哪些模型。
    

第三方 Raycast extension 的源码也显示 Spokenly 的 bundle id 是 `app.spokenly`，偏好路径是 `~/Library/Preferences/app.spokenly.plist`。([GitHub](https://raw.githubusercontent.com/mattiacolombomc/raycast-spokenly/main/src/lib/constants.ts "raw.githubusercontent.com")) 它还说明这些偏好不是普通字符串，而是 JSON 编码后存在 plist 的 Data blob 里，所以不能简单用 `defaults write -string` 改。([GitHub](https://raw.githubusercontent.com/mattiacolombomc/raycast-spokenly/main/src/lib/plist.ts "raw.githubusercontent.com"))

如果你看到：

```text
transcriptionModelID = "你选的新模型"
fileTranscriptionVoiceModelID = "旧模型"
```

那说明 live dictation 配置已经是新的，问题可能是 live fallback / 快捷键入口 / 缓存。

如果你看到：

```text
transcriptionModelID = "旧模型"
fileTranscriptionVoiceModelID = "你选的新模型"
```

那就说明你改到的是 Retry/file transcription 那条路径，不是 live dictation。

## 9. 我对 Spokenly 这件事的判断

最可能的解释是：

> History Retry 是拿保存的音频重新跑一次，所以它读取了你当前选择的新模型；但你平时按快捷键开始的 live dictation 可能走的是另一个 prompt/quick command，或者因为实时模式/模型不可用而 fallback 到旧模型。

最实用的处理顺序是：

1. 选好模型后彻底 Quit / reopen Spokenly。
    
2. 从主界面直接 dictation，不用快捷键，确认是否仍是旧模型。
    
3. 检查 Quick Commands / Main Prompt / Advanced Settings 里有没有旧模型或旧 AI Provider。
    
4. 用上面的 Terminal 命令看 `transcriptionModelID` 和 `fileTranscriptionVoiceModelID` 是否一致。
    
5. 确认目标模型的 API key / 本地下载 / realtime 支持。
    
6. 如果 plist 里 `transcriptionModelID` 已经是新模型，但 live dictation 仍显示旧模型，那基本就是 Spokenly 的 bug 或 fallback 行为，应该把这几个 model ID、你的 Spokenly 版本、录屏发给开发者。
    

---

## 我会给你的最终工作流建议

你的多开方式本身没问题。问题是你现在缺少“可寻址的结构”。

我会这样改：

1. **5 个固定 Spaces**，永远不自动重排。
    
2. **`Ctrl+1..5` 直接跳转**，少用四指上滑。
    
3. **Rectangle 或 Raycast 管窗口**，用快捷键做左右/三分之一布局。
    
4. **AltTab 做窗口级切换**，弥补 macOS `Command+Tab` 只切 app 的问题。
    
5. **Comms 单独一个 Space**，不要让 email/Slack 到处污染。
    
6. **每个任务只保留“主窗口 + 辅助窗口”两栏**，其他信息放到别的 Space。
    
7. **浏览器按项目开 window，不按随机 tab 堆叠**。
    
8. **Hyper key** 做自己的 macOS prefix key，把 Mac 调成 tmux。