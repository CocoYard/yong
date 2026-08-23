---
layout: post
title: 标题（小写，短，不加句号）
date: 2026-08-23 10:00:00
description: 一句话摘要，显示在 blog 列表页和 <meta> 里
tags: blender workflow
categories: notes
# 下面这些按需打开，不用就整行删掉
# thumbnail: assets/img/xxx.jpg  # 列表页缩略图
# giscus_comments: true          # GitHub Discussions 评论
# related_posts: false           # 关掉文末的相关文章
# published: false               # 写完之前设 false，不会构建到线上
---

正文从这里开始，第一段会被当成导语。段落之间要空一行，单纯换行不生效。

## 二级标题

`##` 是正文里最大的标题，不要用 `#`（`#` 留给 front matter 的 title）。

**粗体**、*斜体*、`行内代码`、[链接](https://docs.blender.org/)、脚注[^1]。

[^1]: 脚注写在文章任意位置都行，渲染时自动收到文末。

### 三级标题

- 无序列表
- 第二项
  - 缩进两格是子项

1. 有序列表
2. 第二项

> 引用块，适合放官方文档原话或者踩坑结论。

---

## 代码

围栏加语言名，Rouge 自动高亮，支持 100+ 语言：

```python
import bpy

for obj in bpy.context.selected_objects:
    obj.modifiers.new(name="Subdiv", type="SUBSURF")
```

快捷键这种用行内代码就够了：按 `Ctrl` + `A` 应用变换。

## 图片

图片放 `assets/img/` 下。单张：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/你的图.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    图注写这里，渲染成居中小字。
</div>

并排多张就在同一个 `row` 里放多个 `col-sm`，宽度自动平分。
`zoomable=true` 点击可放大，`rounded z-depth-1` 是圆角和阴影。

## 数学

行内 $$ E = mc^2 $$；独占一行就把 `$$...$$` 单独成段：

$$ \mathbf{n} = \frac{(\mathbf{b}-\mathbf{a}) \times (\mathbf{c}-\mathbf{a})}{\|(\mathbf{b}-\mathbf{a}) \times (\mathbf{c}-\mathbf{a})\|} $$

## 表格

| 快捷键 | 作用 | 备注 |
| :--- | :--- | :--- |
| `Ctrl` + `A` | 应用变换 | 最常忘 |
| `Alt` + `C` | 转换类型 | 3.x 后挪到右键菜单 |

## 视频

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <video class="img-fluid rounded z-depth-1" controls>
            <source src="{{ '/assets/video/xxx.mp4' | relative_url }}" type="video/mp4">
        </video>
    </div>
</div>
