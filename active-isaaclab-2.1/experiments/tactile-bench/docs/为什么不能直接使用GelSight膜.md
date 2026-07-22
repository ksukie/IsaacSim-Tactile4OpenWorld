# 为什么不能直接使用 GelSight 膜

## 结论

不是不能参考 GelSight，而是不能把“GelSight 的膜”直接当成一个现成的 `fxyz` 三维力传感器接进 UIPC。

真实 GelSight 是一个完整触觉成像系统：

```text
硅胶膜 / 弹性体
反光涂层
相机
多方向光源
标定模型
可选 marker dots
```

它直接观测到的通常是：

```text
RGB 图像
表面法线
高度 / 深度
marker 位移
接触形状
```

而不是直接观测到：

```text
每个像素的 fx / fy / fz
```

因此，即使把膜设计成 GelSight-like，仍然必须解决：

```text
1. UIPC 中如何表示这张软膜 mesh
2. 硅胶材料参数如何近似
3. 背面和边缘如何固定
4. 如何从膜形变得到 fxyz
5. 如何把 sim force 标定到真实 Newton
```

## 为什么不能直接拿公开 GelSight / TACTO / Taxim 的膜

公开 GelSight 相关仓库通常更关注：

```text
接触几何 -> RGB tactile image
接触几何 -> depth / normal map
光照模型
marker 渲染或追踪
```

它们一般不直接提供：

```text
可直接导入 UIPC 的 FEM 软膜网格
已标定硅胶本构参数
逐像素三维接触力 fxyz
UIPC contact force 接口
```

所以直接使用这些仓库里的“膜”并不能自动得到我们需要的：

```text
fxyz: T x 300 x 300 x 3
```

我们当前目标不是生成 GelSight 风格 RGB，而是：

```text
UIPC 软膜真实形变
-> dense deformation / constitutive model
-> fx / fy / fz 三维力场
```

## GelSight 能给我们的启发

GelSight 最值得借鉴的不是 RGB 外观，而是观测思想：

```text
整张膜表面被 dense 观测
接触形状来自膜表面形变
切向位移可通过 marker-like tracking 估计
最终力需要材料模型或标定
```

对应到本项目，合理路线是：

```text
GelSight Mini 尺寸
硅胶近似材料
高密度规则前表面 mesh
合理背面 / 边缘固定
dense surface deformation map
可选 marker-like shear tracking
输出 fxyz 而不是 RGB
```

## 当前 V2 与 GelSight-like 方案的区别

当前 V2：

```text
UIPC 表面顶点 surf_nodal_pos_w
-> 顶点 compression / shear displacement
-> 本构力估计
-> conservative splat 到 300 x 300
```

更接近 GelSight 思想的后续版本应升级为：

```text
UIPC 前表面三角面片
-> rasterize 到 300 x 300
-> 每个像素插值 rest/current surface
-> dense compression / shear / contact_mask
-> fxyz
```

这样比单纯顶点 splat 更接近：

```text
整张膜表面形变区域被观测
```

也更适合表达：

```text
凸起
边缘
孔洞
纹理几何
接触轮廓
```

## 为什么即使换成 GelSight-like 膜也不是直接真实力

GelSight 本身主要看到的是膜形状和 marker 位移。力通常仍需要：

```text
材料模型
摩擦模型
外部力计标定
学习模型
真实传感器数据对齐
```

本项目在 V8 之前的力单位仍应写为：

```text
sim_constitutive_force
```

不能宣称为真实 Newton。

## 预期真实程度

如果实现：

```text
尺寸参考 GelSight Mini
硅胶近似材料
高密度规则 mesh
合理固定
dense deformation map
fxyz 输出
```

工程上可以期待：

```text
接触形状真实性：较好
孔洞 / 凸起结构保真：中等到较好
法向 fz 相对分布：中等
切向 fx/fy 趋势：中等偏弱
绝对 Newton 数值：未标定前不可靠
```

更精确的判断必须通过后续验证：

```text
实心球 / 圆柱压膜
有孔压头压膜
横向摩擦
释放回零
真实或仿真力计标定
```

## 推荐路线

不要直接追求“下载一个 GelSight 膜”。更合理的是做一个 GelSight-like UIPC membrane：

```text
V2: API 化当前 deformation -> fxyz 核心
V2.1: OpenWorldTactile dense mesh deformation
V2.2: OpenWorldTactile camera-observed membrane
V3: 验证实心物体连续接触、有孔物体保留孔洞，并选择主路线
V4/V5: 挂载到传感器和插孔流程
V6: 保存 HDF5
V8: 真实力标定
```

核心原则：

```text
GelSight 是设计参考，不是可直接替换的 fxyz 模块。
```
