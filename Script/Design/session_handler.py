from typing import Dict, List, Any
from Script.Core import cache_control, game_type
from Script.Design import constant

cache: game_type.Cache = cache_control.cache

class BaseSessionHandler:
    """ 基础会话处理器接口 """
    
    def __init__(self, session_uid: str):
        self.session_uid = session_uid
        
    @property
    def session(self):
        return cache.interaction_sessions.get(self.session_uid)
        
    def on_start(self):
        """ 会话成功建立时的初始化处理 """
        pass
        
    def on_update(self) -> bool:
        """ 会话每 Tick/每分钟 的演进逻辑, 返回 True 表示结束 """
        return False
        
    def on_member_join(self, character_id: int):
        """ 新成员加入时的处理 """
        pass
        
    def on_member_leave(self, character_id: int):
        """ 成员离开时的处理 """
        pass
        
    def on_finish(self):
        """ 会话结束时的结算处理 """
        pass

class ChatSessionHandler(BaseSessionHandler):
    """ 闲聊会话处理器 """
    
    def on_start(self):
        session = self.session
        if not session:
            return
        session.data.setdefault('stage', 0)
        session.data.setdefault('atmosphere', 50)
        
    def on_update(self):
        import random
        session = self.session
        if not session:
            return True
            
        elapsed_time = cache.game_time - session.start_time
        # 每5分钟推进一个阶段
        if elapsed_time > 5 * 60:
            stage = session.data.get('stage', 0)
            if stage == 0:
                session.data['atmosphere'] += random.randint(-10, 20)
                session.data['stage'] = 1
                session.start_time = cache.game_time # 重置计时
            elif stage == 1:
                if session.data['atmosphere'] > 60:
                    self._settle_chat_effect()
                session.data['atmosphere'] += random.randint(-10, 20)
                session.data['stage'] = 2
                session.start_time = cache.game_time
            elif stage >= 2:
                return True # 返回 True 表示会话应结束
        return False
        
    def _settle_chat_effect(self):
        """ 闲聊阶段结算效果 """
        session = self.session
        if not session:
            return
        for member_id in session.members:
            member_data = cache.character_data.get(member_id)
            if member_data:
                member_data.status.setdefault(25, 0)
                member_data.status[25] = max(0, member_data.status[25] - 1) # Reduce fatigue
                
    def on_finish(self):
        # 闲聊结束后的清理
        pass

# 会话处理器工厂/注册表
SESSION_HANDLERS = {
    constant.Behavior.CHAT: ChatSessionHandler,
    # 后续可注册更多的 Handler
}

def get_session_handler(session_uid: str):
    session = cache.interaction_sessions.get(session_uid)
    if not session:
        return None
    handler_class = SESSION_HANDLERS.get(session.type, BaseSessionHandler)
    return handler_class(session_uid)
