import logging
import os
import threading

class Logger:
    _instance = None  # 保存单例实例
    _lock = threading.Lock()  # 用于多线程锁

    def __new__(cls, log_file=None, log_level=logging.WARNING):
        """确保只创建一个实例，并通过加锁保证线程安全"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # 双重检查锁
                    cls._instance = super(Logger, cls).__new__(cls)
                    cls._instance.init_logger(log_file, log_level)
        return cls._instance

    def init_logger(self, log_file, log_level):
        """初始化 logger 实例"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # 创建日志格式化器
        formatter = logging.Formatter(
            # '%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d',
            '[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 防止重复添加处理器
        if not self.logger.hasHandlers():
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # 创建文件处理器
            if log_file is None:
                log_file = os.path.join(os.path.dirname(__file__), 'app.log')
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self):
        """返回 logger 实例"""
        return self.logger
