import flet
from flet.canvas import Canvas, Path
from flet.core.painting import Paint, PaintingStyle
from flet.core.types import StrokeCap
from slot import Slot

class Connection:
    """节点之间的连接线"""
    def __init__(
        self,
        from_slot: Slot,
        to_slot: Slot,
        canvas: Canvas,
        color: str = flet.colors.GREEN,
    ):
        self.from_slot = from_slot
        self.to_slot = to_slot
        self.canvas = canvas
        self.color = color
        
        # 添加连接线的样式
        self.paint = Paint(
            stroke_width=2,
            color=flet.colors.BLUE,
            style=PaintingStyle.STROKE,
            stroke_cap=StrokeCap.ROUND,
        )
        
        # 创建连接线路径
        self.path = None
        self.update()
        
    def update(self):
        """更新连接线的位置"""
        # 获取起点和终点
        start_pos = self.from_slot.get_center_position()
        end_pos = self.to_slot.get_center_position()
        
        # 计算控制点（创建平滑的贝塞尔曲线）
        dx = end_pos[0] - start_pos[0]
        control_x1 = start_pos[0] + dx * 0.5
        control_x2 = end_pos[0] - dx * 0.5
        
        # 创建路径
        path = Path([
            Path.MoveTo(start_pos[0], start_pos[1]),
            Path.CubicTo(
                control_x1, start_pos[1],  # 第一个控制点
                control_x2, end_pos[1],    # 第二个控制点
                end_pos[0], end_pos[1]     # 终点
            )
        ])
        
        # 设置路径样式
        path.paint = Paint(
            stroke_width=2,
            color=self.color,  # 使用实例的颜色属性
            style=PaintingStyle.STROKE,
            stroke_cap=StrokeCap.ROUND,
        )
        
        # 更新画布
        if self.path in self.canvas.shapes:
            self.canvas.shapes.remove(self.path)
        self.path = path
        self.canvas.shapes.append(self.path)
        self.canvas.update()
    
    def remove(self):
        """从画布中移除连接线"""
        if self.path in self.canvas.shapes:
            self.canvas.shapes.remove(self.path)
            self.canvas.update()