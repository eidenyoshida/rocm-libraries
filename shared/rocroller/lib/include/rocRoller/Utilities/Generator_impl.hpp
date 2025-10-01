/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <string>

#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    inline std::string toString(GeneratorState s)
    {
        switch(s)
        {
        case GeneratorState::NoValue:
            return "NoValue";
        case GeneratorState::HasValue:
            return "HasValue";
        case GeneratorState::HasRange:
            return "HasRange";
        case GeneratorState::HasRangeValue:
            return "HasRangeValue";
        case GeneratorState::HasCopiedValue:
            return "HasCopiedValue";
        case GeneratorState::Done:
            return "Done";

        case GeneratorState::Count:
        default:
            break;
        }
        throw std::runtime_error("Invalid GeneratorState");
    }

    inline std::ostream& operator<<(std::ostream& stream, GeneratorState const& s)
    {
        return stream << toString(s);
    }

    template <typename T, CInputRangeOf<T> TheRange>
    template <std::convertible_to<TheRange> ARange>
    ConcreteRange<T, TheRange>::ConcreteRange(ARange&& r)
        : m_range(std::forward<ARange>(r))
        , m_iter(std::begin(m_range))
    {
    }

    template <typename T, CInputRangeOf<T> TheRange>
    constexpr std::optional<T> ConcreteRange<T, TheRange>::take_value()
    {
        if(m_iter == m_range.end())
            return {};
        return *m_iter;
    }

    template <typename T, CInputRangeOf<T> TheRange>
    constexpr void ConcreteRange<T, TheRange>::increment()
    {
        if(m_iter != m_range.end())
            ++m_iter;
    }

    template <std::movable T>
    Generator<T> Generator<T>::promise_type::get_return_object()
    {
        return Generator<T>{Handle::from_promise(*this)};
    }

    template <std::movable T>
    constexpr void Generator<T>::promise_type::unhandled_exception() noexcept
    {
        m_exception = std::current_exception();
        m_value.reset();
        m_coroutine = nullptr;
    }

    template <std::movable T>
    inline void Generator<T>::promise_type::check_exception() const
    {
        // NOTE: A thread-safe version of this would be similar to
        //     std::exception_ptr exc = nullptr;
        //     std::swap(exc, m_exception);
        //     if(exc)
        //       std::rethrow_exception(exc);
        //
        // Thread-safety is not necessary (RR generates within a single thread); and
        // conditionally swapping improves performance.
        if(m_exception)
        {
            std::exception_ptr exc = nullptr;
            std::swap(exc, m_exception);
            std::rethrow_exception(exc);
        }
    }

    template <std::movable T>
    void Generator<T>::promise_type::advance()
    {
        m_value.reset();
        if(m_coroutine)
        {
            auto& promise = m_coroutine.promise();
            promise.advance();
            m_value = promise.value();

            if(not m_value)
            {
                m_coroutine = nullptr;
            }
        }

        if(m_value)
            return;

        auto handle = std::coroutine_handle<promise_type>::from_promise(*this);
        if(not handle.done())
            handle.resume();
    }

    template <std::movable T>
    constexpr std::suspend_always Generator<T>::promise_type::yield_value(T v) noexcept
    {
        m_value     = std::move(v);
        m_coroutine = nullptr;

        return {};
    }

    template <std::movable T>
    template <CInputRangeOf<T> ARange>
    auto Generator<T>::promise_type::yield_value(ARange&& r) noexcept
    {
        auto gen = [](auto iter, auto E) -> Generator<T> {
            for(; iter != E; iter++)
            {
                co_yield *iter;
            }
        };

        return yield_value(gen(r.begin(), r.end()));
    }

    template <std::movable T>
    auto Generator<T>::promise_type::yield_value(Generator<T>&& r) noexcept
    {
        struct Awaitable
        {
            Generator<T> gen;

            Awaitable(Generator<T>&& gen)
                : gen(std::move(gen))
            {
            }

            bool ready = false;

            bool await_ready() noexcept
            {
                return ready;
            }

            void await_suspend(std::coroutine_handle<promise_type>) noexcept {}

            void await_resume() {}
        };

        m_value.reset();

        Awaitable awaitable{std::move(r)};

        try
        {
            // The C++ runtime will not catch an exception from here, so we
            // must handle it ourselves.

            m_coroutine = awaitable.gen.m_coroutine;

            m_coroutine.resume();
            m_value = m_coroutine.promise().value();

            if(m_value)
            {
                awaitable.ready = false;
            }
            else
            {
                m_coroutine     = nullptr;
                awaitable.ready = true;
            }
        }
        catch(...)
        {
            unhandled_exception();
        }

        return awaitable;
    }

    template <std::movable T>
    auto Generator<T>::promise_type::yield_value(std::initializer_list<T> r) noexcept
    {
        return yield_value<std::initializer_list<T>>(std::move(r));
    }

    template <std::movable T>
    constexpr std::optional<T> const& Generator<T>::promise_type::value() const
    {
        check_exception();

        return m_value;
    }

    template <std::movable T>
    void Generator<T>::promise_type::discard_value()
    {
        check_exception();

        if(m_value.has_value())
            m_value.reset();
    }

    // pre-increment
    template <std::movable T>
    auto Generator<T>::Iterator::operator++() -> Iterator&
    {
        auto& promise = m_coroutine.promise();
        promise.check_exception();
        promise.advance();
        return *this;
    }

    // post-increment
    template <std::movable T>
    void Generator<T>::Iterator::operator++(int)
    {
        ++(*this);
    }

    template <std::movable T>
    T const& Generator<T>::Iterator::operator*() const
    {
        auto const& value = m_coroutine.promise().value();
        if(not value)
            throw std::runtime_error("operator* is nullptr!");
        return *value;
    }

    template <std::movable T>
    T const* Generator<T>::Iterator::operator->() const
    {
        return &(*m_coroutine.promise().value());
    }

    template <std::movable T>
    bool Generator<T>::Iterator::isDone() const
    {
        if(not m_coroutine)
            return true;

        m_coroutine.promise().check_exception();

        if(not m_coroutine.done())
            return false;

        return true;
    }

    template <std::movable T>
    bool Generator<T>::Iterator::operator==(std::default_sentinel_t) const
    {
        return isDone();
    }

    template <std::movable T>
    bool Generator<T>::Iterator::operator!=(std::default_sentinel_t t) const
    {
        return !(*this == t);
    }

    template <std::movable T>
    Generator<T>::Iterator::Iterator(Handle const& coroutine)
        : m_coroutine{coroutine}
    {
    }

    template <std::movable T>
    Generator<T>::Iterator::Iterator()
    {
    }

    template <std::movable T>
    Generator<T>::Iterator::Iterator(T value)
    {
    }

    template <std::movable T>
    Generator<T>::Generator(Handle const& coroutine)
        : m_coroutine{coroutine}
    {
    }

    template <std::movable T>
    Generator<T>::~Generator()
    {
        if(m_coroutine)
        {
            m_coroutine.destroy();
        }
    }

    template <std::movable T>
    Generator<T>::Generator(Generator&& other) noexcept
        : m_coroutine{other.m_coroutine}
    {
        other.m_coroutine = {};
    }

    template <std::movable T>
    auto Generator<T>::operator=(Generator&& rhs) noexcept -> Generator&
    {
        if(this != &rhs)
        {
            std::swap(m_coroutine, rhs.m_coroutine);
        }
        return *this;
    }

    template <std::movable T>
    constexpr auto Generator<T>::begin() -> iterator
    {
        m_coroutine.promise().advance();
        return iterator{m_coroutine};
    }

    template <std::movable T>
    constexpr auto Generator<T>::end() -> iterator
    {
        return iterator{std::default_sentinel_t{}};
    }

    template <std::movable T>
    template <template <typename...> typename Container>
    constexpr Container<T> Generator<T>::to()
    {
        auto b = begin();
        auto e = end();
        return Container<T>(b, e);
    }

    template <std::movable T>
    template <std::predicate<T> Predicate>
    Generator<T> Generator<T>::filter(Predicate predicate)
    {
        return rocRoller::filter(predicate, std::move(*this));
    }

    template <std::movable T>
    template <std::invocable<T> Func>
    Generator<std::invoke_result_t<Func, T>> Generator<T>::map(Func func)
    {
        return rocRoller::map(func, std::move(*this));
    }

    template <std::movable T>
    Generator<T> Generator<T>::take(size_t n)
    {
        return rocRoller::take(n, std::move(*this));
    }

    template <std::ranges::input_range Range, typename Predicate>
    requires(std::predicate<Predicate, std::ranges::range_value_t<Range>>)
        Generator<std::ranges::range_value_t<Range>> filter(Predicate predicate, Range range)
    {
        for(auto val : range)
        {
            if(predicate(val))
                co_yield val;
        }
    }

    template <std::ranges::input_range Range, typename Func>
    requires(std::invocable<Func, std::ranges::range_value_t<Range>>)
        Generator<std::invoke_result_t<Func, std::ranges::range_value_t<Range>>> map(Func  func,
                                                                                     Range range)
    {
        for(auto val : range)
            co_yield func(val);
    }

    template <typename T>
    Generator<T> take(size_t n, Generator<T> gen)
    {
        auto it = gen.begin();
        for(size_t i = 0; i < n && it != gen.end(); ++i, ++it)
        {
            co_yield *it;
        }
    }

    template <std::movable T>
    std::optional<T> Generator<T>::only()
    {
        return rocRoller::only(std::move(*this));
    }

    template <std::ranges::input_range Range>
    inline std::optional<std::ranges::range_value_t<Range>> only(Range range)
    {
        auto iter = range.begin();
        if(iter == range.end())
            return {};

        auto value = *iter;

        ++iter;
        if(iter == range.end())
            return value;

        return {};
    }

    template <std::movable T>
    constexpr bool Generator<T>::empty()
    {
        return begin() == end();
    }

    template <std::ranges::input_range Range>
    inline constexpr bool empty(Range range)
    {
        return range.begin() == range.end();
    }

}
