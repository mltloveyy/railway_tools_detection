## 手机端

### 接口描述

*   `接口功能:` 接收图像路径，执行目标检测，输出检测结果文件路径

*   `调用方式:` 异步回调

*   `性能要求:` 单图处理延迟（含I/O操作）

*   `依赖框架:` MNN 2.6.0

### 接口函数定义

#### 回调

```java
public interface AlgoCallback<T> {
    void onSuccess(T result);      // 成功时返回结果文件路径
    void onFailure(String error);  // 失败时返回错误信息
}
```

#### 检测器

```java
public class Detector{

    /**
     * 预加载模型
     * @param modelPath 必填 模型文件路径
     * @param callback      返回模型ID（用于后续操作）
     */
    public static void preload(
        String modelPath, 
        AlgoCallback<String> callback
    );

    /**
     * 执行推理
     * @param modelId    必填 预加载的模型ID
     * @param imagePaths 必填 输入图像路径列表
     * @param outputDir  必填 结果文件保存目录
     * @param callback        返回结果文件路径列表
     */
    public static void run(
        String modelId,
        List<String> imagePaths,
        File outputDir,
        AlgoCallback<List<String>> callback
    );

    /**
     * 释放模型资源
     * @param modelId  必填 预加载的模型ID
     * @param callback      返回释放状态
     */
    public static void release(
        String modelId, 
        AlgoCallback<Void> callback
    );
}
```

#### 标注器

```java
public class Annotation{
    private final List<Point> points;

    public Annotation(List<Point> points) {
        this.points = points;
    }

    /**
     * 轮廓点
     * @param points 返回轮廓点
     */
    public List<Point> getPoints() {
        return points();
    }

    /**
     * 外接矩形
     * @param rect 返回外接矩形端点
     */
    public List<Point> getRect() {
        return toRect();
    }
}

public class Annotator{

    /**
     * 预加载模型
     * @param modelPath 必填  模型文件路径
     * @param callback       返回模型ID（用于后续操作）
     */
    public static void preload(
        String modelPath, 
        AlgoCallback<String> callback
    );

    /**
     * 加载图像
     * @param modelId   必填 预加载的模型ID
     * @param imagePath 必填 输入图像路径
     * @param callback       返回图像加载状态
     */
    public static void setImage(
        String modelId,
        String imagePath,
        AlgoCallback<String> callback
    );

    /**
     * 执行推理
     * @param modelId   必填 预加载的模型ID
     * @param points    必填 输入坐标点列表
     * @param labels    必填 输入坐标点类型列表
     * @param callback       返回标注掩膜
     */
    public static void run(
        String modelId,
        List<Point> points,
        List<Int> labels,
        AlgoCallback<Annotation> callback
    );

    /**
     * 释放模型资源
     * @param modelId  必填 预加载的模型ID
     * @param callback     返回释放状态
     */
    public static void release(
        String modelId, 
        AlgoCallback<Void> callback
    );
}
```

### 标注数据格式示例

```json
{
  "version": "5.5.0",
  "flags": {},
  "shapes": [
    {
      "label": "xiezuiqian",
      "points": [
        [
          271.9234234234234,
          121.21621621621621
        ],
        [
          98.95045045045043,
          251.84684684684686
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    },
    {
      "label": "jianzuiqian",
      "points": [
        [
          369.2207207207207,
          103.1981981981982
        ],
        [
          206.15765765765764,
          242.83783783783784
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    },
  ],
  "imagePath": "/sdcard/DCIM/IMG_20250425.jpg",
  "imageData": null,
  "imageHeight": 1920,
  "imageWidth": 1080
}
```

#### 有效字段说明

| 字段          | 类型     | 描述              |
| ----------- | ------ | --------------- |
| version     | string | 标注文件版本，固定为5.5.0 |
| shapes      | /      | 包含所有的检测结果       |
| label       | string | 工具名称            |
| points      | float  | 矩形右上角和左下角的两个端点  |
| shape\_type | string | 必须为"rectangle"  |
| imagePath   | string | 图片绝对路径          |
| imageHeight | int    | 图像高度            |
| imageWidth  | int    | 图像宽度            |

### 异常处理

| 错误码  | 描述                | 处理建议             |
| ---- | ----------------- | ---------------- |
| 1001 | 文件路径无效            | 检查存储权限和文件存在性     |
| 1002 | 模型未初始化            | 调用模型预加载接口preload |
| 1003 | 内存不足(OOM)         | 检查内存占用情况，清除无效进程  |
| 1020 | 标注模型未加载图像         | 调用图像加载接口setImage |
| 1021 | 坐标点列表与坐标点标签列表长度不同 | 检查输入列表长度         |

## 服务器

### 接口描述

*   `接口功能:` 接收训练数据路径和模型保存路径，执行模型训练，保存模型到指定路径下

*   `调用方式:` PowerShell指令

*   `依赖框架:` PyTorch 2.5.1 (带英伟达显卡：PyTorch-cuda 2.5.1)

### 调用示例

```powershell
python train.py --dataset ${训练数据路径} --output ${模型保存路径}
```

