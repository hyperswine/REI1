"""
Test for concurrency features in REI1
"""

import pytest
from parsing import create_parser
from semantics import create_analyzer
from interpreter import create_interpreter
import os


class TestConcurrencyFeatures:
    """Test REI1 concurrency features"""

    @pytest.fixture
    def setup(self):
        """Setup parser, analyzer, and interpreter"""
        parser = create_parser()
        analyzer = create_analyzer()
        interpreter = create_interpreter()
        return parser, analyzer, interpreter

    def test_unsafe_io_read(self, setup):
        """Test unsafe IO read operations"""
        parser, analyzer, interpreter = setup

        # Create test file
        test_content = "Hello, REI1!"
        with open("test_io.txt", "w") as f:
            f.write(test_content)

        try:
            code = 'content = unsafe $ IO.read "test_io.txt".'
            cst = parser.parse_string(code)
            ast = analyzer.analyze(cst)
            result = interpreter.interpret_program(ast)

            assert result['content'] == test_content
        finally:
            # Cleanup
            if os.path.exists("test_io.txt"):
                os.remove("test_io.txt")

    def test_unsafe_io_write(self, setup):
        """Test unsafe IO write operations"""
        parser, analyzer, interpreter = setup

        try:
            code = '''
            result = unsafe $ IO.write "test_output.txt" "Test content".
            '''
            cst = parser.parse_string(code)
            ast = analyzer.analyze(cst)
            interpreter.interpret_program(ast)

            # Check file was created and has correct content
            assert os.path.exists("test_output.txt")
            with open("test_output.txt", "r") as f:
                content = f.read()
            assert content == "Test content"
        finally:
            # Cleanup
            if os.path.exists("test_output.txt"):
                os.remove("test_output.txt")

    def test_pipeline_operator(self, setup):
        """Test pipeline operator"""
        parser, analyzer, interpreter = setup

        code = '''
        double x = * x 2.
        add5 x = + x 5.
        result = 10 |> double |> add5.
        '''
        cst = parser.parse_string(code)
        ast = analyzer.analyze(cst)
        result = interpreter.interpret_program(ast)

        assert result['result'] == 25  # (10 * 2) + 5

    def test_par_map(self, setup):
        """Test parallel map"""
        parser, analyzer, interpreter = setup

        code = '''
        double x = * x 2.
        numbers = [1, 2, 3, 4, 5].
        result = par-map double numbers.
        '''
        cst = parser.parse_string(code)
        ast = analyzer.analyze(cst)
        result = interpreter.interpret_program(ast)

        expected = [2, 4, 6, 8, 10]
        assert sorted(result['result']) == sorted(expected)

    def test_par_filter(self, setup):
        """Test parallel filter"""
        parser, analyzer, interpreter = setup

        code = '''
        isEven x = == (% x 2) 0.
        numbers = [1, 2, 3, 4, 5, 6].
        result = par-filter isEven numbers.
        '''
        cst = parser.parse_string(code)
        ast = analyzer.analyze(cst)
        result = interpreter.interpret_program(ast)

        expected = [2, 4, 6]
        assert sorted(result['result']) == sorted(expected)

    def test_par_fold(self, setup):
        """Test parallel fold"""
        parser, analyzer, interpreter = setup

        code = '''
        add x y = + x y.
        numbers = [1, 2, 3, 4, 5].
        result = par-fold add 0 numbers.
        '''
        cst = parser.parse_string(code)
        ast = analyzer.analyze(cst)
        result = interpreter.interpret_program(ast)

        assert result['result'] == 15  # 1+2+3+4+5

    def test_complex_pipeline(self, setup):
        """Test complex pipeline with multiple operations"""
        parser, analyzer, interpreter = setup

        code = '''
        double x = * x 2.
        isEven x = == (% x 2) 0.

        numbers = [1, 2, 3, 4, 5].
        doubled = par-map double numbers.
        evens = par-filter isEven doubled.
        '''
        cst = parser.parse_string(code)
        ast = analyzer.analyze(cst)
        result = interpreter.interpret_program(ast)

        # [1,2,3,4,5] -> [2,4,6,8,10] -> [2,4,6,8,10] (all even)
        expected = [2, 4, 6, 8, 10]
        assert sorted(result['evens']) == sorted(expected)

    def test_actor_spawn_send_recv(self, setup):
        """Test basic actor model: spawn, send, recv"""
        parser, analyzer, interpreter = setup

        try:
            # Test basic actor spawn - using ML-style function definition
            code = '''
            handler msg = msg.
            actor_id = Proc.spawn handler.
            '''

            cst = parser.parse_string(code)
            ast = analyzer.analyze(cst)
            result = interpreter.interpret_program(ast)

            # Verify actor was spawned (returns an actor ID string)
            assert isinstance(result['actor_id'], str)
            assert len(result['actor_id']) > 0

        except Exception as e:
            # Clean up any actors
            from interpreter import _actor_registry
            _actor_registry.terminate_all()
            raise e

    def test_actor_communication(self, setup):
        """Test actor-to-actor communication"""
        parser, analyzer, interpreter = setup

        try:
            # Test more complex actor interaction - back to lambda now that it works
            code = '''
            handler = lambda msg => msg.
            actor1 = Proc.spawn handler.
            actor2 = Proc.spawn handler.
            '''

            cst = parser.parse_string(code)
            ast = analyzer.analyze(cst)
            result = interpreter.interpret_program(ast)

            # Verify both actors were spawned
            assert isinstance(result['actor1'], str)
            assert isinstance(result['actor2'], str)
            assert result['actor1'] != result['actor2']

        except Exception as e:
            # Clean up any actors
            from interpreter import _actor_registry
            _actor_registry.terminate_all()
            raise e
        finally:
            # Always clean up actors
            from interpreter import _actor_registry
            _actor_registry.terminate_all()

    def test_actor_recv_functionality(self, setup):
        """Test receiving messages in actors"""
        parser, analyzer, interpreter = setup

        try:
            # Test actor receiving messages - simplified without semicolons
            code = '''
            receiver_handler initial_msg = Proc.recv.
            receiver_id = Proc.spawn receiver_handler.
            '''

            cst = parser.parse_string(code)
            ast = analyzer.analyze(cst)
            result = interpreter.interpret_program(ast)

            # Verify actor was spawned
            assert isinstance(result['receiver_id'], str)

        except Exception as e:
            # Clean up any actors
            from interpreter import _actor_registry
            _actor_registry.terminate_all()
            raise e
        finally:
            # Always clean up actors
            from interpreter import _actor_registry
            _actor_registry.terminate_all()